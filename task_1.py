import datetime
import pickle
from functools import partial
from glob import glob

import dateutil
import pandas as pd
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from common import preprocess
from ctfidf import CTFIDFVectorizer

model = SentenceTransformer('all-roberta-large-v1')


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


def rank_topics(data):
    topics = data.groupby(['topic'], as_index=False).agg({' body': ' '.join})
    topics['count'] = data.groupby(['topic']).count()['title'].tolist()
    return topics


def extract_topics_ctfidf(data):
    topics = rank_topics(data)

    count_vectorizer = CountVectorizer(ngram_range=(1, 3), preprocessor=partial(preprocess, stem_lemmatize=False))
    count = count_vectorizer.fit_transform(topics[' body'])
    words = count_vectorizer.get_feature_names_out()

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

    return topics


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


def get_topics_year(data):
    return {
        2015: extract_topics_ctfidf(data[data.year == 2015]),
        2016: extract_topics_ctfidf(data[data.year == 2016]),
        2017: extract_topics_ctfidf(data[data.year == 2017]),
    }


if __name__ == '__main__':
    data = get_data()
    embeddings = embed_documents(data)
    cluster_topics(data, embeddings)
    topics_year = get_topics_year(data)
    for year in [2015, 2016, 2017]:
        topics = list(map(' '.join, topics_year[year].sort_values('count', ascending=False)['keyword'].tolist()))
        print(year, ':', ', '.join(topics))
