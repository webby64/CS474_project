import pickle
from glob import glob

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


def extract_topics(data):
    topic = data.groupby(['topic'], as_index=False).agg({' body': ' '.join})
    topic['count'] = data.groupby(['topic']).count()['title'].tolist()

    count_vectorizer = CountVectorizer(ngram_range=(1, 3), preprocessor=lambda x: preprocess(x, False))
    count = count_vectorizer.fit_transform(topic[' body'])
    words = count_vectorizer.get_feature_names_out()

    ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=data.shape[0]).toarray()

    keywords = []
    for label in topic.index:
        candidate = [words[index] for index in ctfidf[label].argsort()[-10:]]
        keyword = []
        for word in candidate:
            for target in candidate:
                if word in target and word != target:
                    break
            else:
                keyword.append(word)
        keywords.append(keyword)
    topic['keyword'] = keywords

    return topic


def get_data_year():
    try:
        with open('results/data_year.pickle', 'rb') as f:
            data_year = pickle.load(f)
    except FileNotFoundError:
        data = pd.concat(map(pd.read_json, glob('data/*.json'))).reset_index(drop=True)

        embeddings = embed_documents(data)
        cluster_topics(data, embeddings)

        data_year = {
            2015: data[(data[' time'] > '2015-01-01') & (data[' time'] < '2016-01-01')],
            2016: data[(data[' time'] > '2016-01-01') & (data[' time'] < '2017-01-01')],
            2017: data[(data[' time'] > '2017-01-01') & (data[' time'] < '2018-01-01')],
        }

        with open('results/data_year.pickle', 'wb') as f:
            pickle.dump(data_year, f)

    return data_year


def get_topics_year():
    try:
        with open('results/topics_year.pickle', 'rb') as f:
            topics_year = pickle.load(f)
    except FileNotFoundError:
        data_year = get_data_year()

        topics_year = {
            2015: extract_topics(data_year[2015]),
            2016: extract_topics(data_year[2016]),
            2017: extract_topics(data_year[2017]),
        }

        with open('results/topics_year.pickle', 'wb') as f:
            pickle.dump(topics_year, f)

    return topics_year


if __name__ == '__main__':
    topics_year = get_topics_year()
    for year in [2015, 2016, 2017]:
        topics = list(map(' '.join, topics_year[year].sort_values('count', ascending=False)['keyword'].tolist()))
        print(year, ':', ', '.join(topics))
