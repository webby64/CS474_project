import pickle
import re
from glob import glob

import nltk
import pandas as pd
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from ctfidf import CTFIDFVectorizer


def embed_documents(data):
    return SentenceTransformer('all-roberta-large-v1').encode(data[' body'].tolist(), show_progress_bar=True)


def cluster_topics(data, embeddings):
    dim_reduction = UMAP(n_components=5, min_dist=0, metric='cosine').fit_transform(embeddings)

    clusterer = HDBSCAN(min_cluster_size=40)
    clusterer.fit(dim_reduction)
    data['topic'] = clusterer.labels_
    data['prob'] = clusterer.probabilities_


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

contractions = {
    "aren't": "are not", "can't": "cannot", "couldn't": "could not", "could've": "could have", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "i'd": "I would",
    "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not", "would've": "would have",
    "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
    "mightn't": "might not", "mayn't": "may not", "might've": "might have", "needn't": "need not",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would", "she'll": "she will",
    "she's": "she is", "shouldn't": "should not", "should've": "should have", "that's": "that is",
    "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "we'd": "we would", "we're": "we are", "weren't": "were not",
    "we've": "we have", "what'll": "what will", "what're": "what are", "what's": "what is",
    "what've": "what have", "where's": "where is", "who'd": "who would", "who'll": "who will",
    "who're": "who are", "who's": "who is", "who've": "who have", "won't": "will not", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have", "wasn't": "was not",
    "we'll": " will", "y'all": "you all", "y'all'd": "you all would", "y'all're": "you all are"
}


def preprocess(text):
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', text)  # clean url
    text = re.sub(r'#(\w+)', '', text)  # clean hashtags
    text = re.sub(r'@(\w+)', '', text)  # clean @s
    text = re.sub(r'<[^>]+>', '', text)  # clean tags
    text = re.sub(r'\d+', '', text)  # clean digits
    text = re.sub(r'’', '\'', text)  # replace ’ with '
    text = re.sub(r's\'', '', text)  # clean s'
    text = re.sub(r'[£₹$€₩]', ' ', text)  # clean currency symbols
    text = re.sub(r'[δ∫βωσ∈∆≡απθ+*-=°^×√÷]', ' ', text)  # clean math symbols
    text = re.sub(r'[/(),!@"“”?.%_&#:;><{}~\[\]|…]', ' ', text)  # clean punctuation
    text = [contractions[word] if word in contractions else word for word in
            text.split()]  # change contractions to full forms
    temp = []
    for word in text:
        if word not in stop_words:
            temp.append(word)
    text = temp
    text = " ".join(text)
    text = re.sub(r'\'s', '', text)  # clean 's
    text = re.sub(r'\'', '', text)  # clean '
    text = re.sub(r'yonhap', '', text)  # clean yonhap
    return text


def extract_topics(data):
    topic = data.groupby(['topic'], as_index=False).agg({' body': ' '.join})
    topic['count'] = data.groupby(['topic']).count()['title'].tolist()

    count_vectorizer = CountVectorizer(ngram_range=(1, 3), preprocessor=preprocess)
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


try:
    with open('results/data_year.pickle', 'rb') as f:
        data_year = pickle.load(f)
except FileNotFoundError:
    data_all = pd.concat(map(pd.read_json, glob('data/*.json'))).reset_index(drop=True)

    embeddings = embed_documents(data_all)

    cluster_topics(data_all, embeddings)

    data_year = {
        2015: data_all[(data_all[' time'] > '2015-01-01') & (data_all[' time'] < '2016-01-01')],
        2016: data_all[(data_all[' time'] > '2016-01-01') & (data_all[' time'] < '2017-01-01')],
        2017: data_all[(data_all[' time'] > '2017-01-01') & (data_all[' time'] < '2018-01-01')],
    }

    with open('results/data_year.pickle', 'wb') as f:
        pickle.dump(data_year, f)

try:
    with open('results/topics_year.pickle', 'rb') as f:
        topics_year = pickle.load(f)
except FileNotFoundError:
    topics_year = {
        2015: extract_topics(data_year[2015]),
        2016: extract_topics(data_year[2016]),
        2017: extract_topics(data_year[2017]),
    }

    with open('results/topics_year.pickle', 'wb') as f:
        pickle.dump(topics_year, f)

for year in [2015, 2016, 2017]:
    topics = list(map(' '.join, topics_year[year].sort_values('count', ascending=False)['keyword'].tolist()))
    print(year, ':', ', '.join(topics))
