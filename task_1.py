import datetime
import pickle
from functools import partial
from glob import glob

import dateutil
import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from common import preprocess
from ctfidf import CTFIDFVectorizer

model = SentenceTransformer('all-roberta-large-v1')


def get_data():
    try:
        with open('results/data.pickle', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = pd.concat(map(pd.read_json, glob('data/*.json'))).reset_index(drop=True)
        data = data.drop(columns=[' author', ' description', ' section'])
        data['year'] = data[' time'].apply(lambda x: int(x.split('-')[0]))
        data['days'] = data.apply(
            lambda x: (dateutil.parser.isoparse(x[' time']).date() - datetime.date(x.year, 1, 1)).days,
            axis=1)

        with open('results/data.pickle', 'wb') as f:
            pickle.dump(data, f)

    return data


def embed_documents(data):
    try:
        with open('results/embeddings.pickle', 'rb') as f:
            embeddings = pickle.load(f)
    except FileNotFoundError:
        embeddings = model.encode(data[' body'].tolist(), show_progress_bar=True)

        with open('results/embeddings.pickle', 'wb') as f:
            pickle.dump(embeddings, f)

    return embeddings


def cluster_topics(data, embeddings):
    dim_reduction = UMAP(n_components=5, min_dist=0, metric='cosine').fit_transform(embeddings)

    clusterer = HDBSCAN(min_cluster_size=40)
    clusterer.fit(dim_reduction)
    data['topic'] = clusterer.labels_
    data['prob'] = clusterer.probabilities_


def max_subset_sum(data, length):
    return max([sum(data[i:i + length]) for i in range(len(data) - length)])


def max_subset_sums_for_topics(data, topics, length):
    max_subset_sums = []
    for i, row in topics.iterrows():
        count = np.zeros(366, dtype=np.int32)
        for day in data[data.topic == row.topic].days:
            count[day] += 1
        max_subset_sums.append(max_subset_sum(count, length))
    return max_subset_sums


def extract_topics(data, top=10, criteria='max_subset_sum', length=30, ctfidf=False):
    group = data.groupby('topic', as_index=False)
    topics = group.agg({' body': ' '.join})

    if criteria == 'count':
        topics['count'] = group.count()[' body'].tolist()
    else:
        criteria = 'max_subset_sum'
        topics['max_subset_sum'] = max_subset_sums_for_topics(data, topics, length)

    if not ctfidf:
        topics = topics.head(top)

    count_vectorizer = CountVectorizer(ngram_range=(1, 3), preprocessor=partial(preprocess, stem_lemmatize=False))
    count = count_vectorizer.fit_transform(topics[' body'])
    words = count_vectorizer.get_feature_names_out()

    if ctfidf:
        ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=data.shape[0]).toarray()

        keywords = []
        for label in topics.index:
            candidate = [words[index] for index in ctfidf[label].argsort()[-10:]]
            keyword = []
            for word in candidate:
                for target in candidate:
                    if word in target and word != target:
                        break
                else:
                    keyword.append(word)
            keywords.append(keyword)
        topics['keyword'] = keywords

        return topics[topics.topic != -1].sort_values(criteria, ascending=False).head(top)
    else:
        # TODO: roberta based
        pass


def get_topics_year(data, **kwargs):
    return {
        2015: extract_topics(data[data.year == 2015], **kwargs),
        2016: extract_topics(data[data.year == 2016], **kwargs),
        2017: extract_topics(data[data.year == 2017], **kwargs),
    }


if __name__ == '__main__':
    data = get_data()
    embeddings = embed_documents(data)
    cluster_topics(data, embeddings)
    topics_year = get_topics_year(data, ctfidf=True)
    for year in [2015, 2016, 2017]:
        print(year)
        print('\n'.join(topics_year[year].keyword.head(10).apply(' '.join).apply(lambda x: '  ' + x)))
