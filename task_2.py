import collections
import json
import os
import re
import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import stanza
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from stanza.server import CoreNLPClient
from wordcloud import WordCloud, STOPWORDS

from task_1 import get_topics_year

corenlp_dir = './corenlp'
os.environ["CORENLP_HOME"] = corenlp_dir

nltk.download('wordnet')
nltk.download('stopwords')


root = './'

'''
File structure

Text Mining
    |--TM.ipynb
    |--data
        |--koreaherald_1517_0.json
        |--koreaherald_1517_1.json
        |--koreaherald_1517_2.json
'''

frames = []
for x in range(8):
  with open(os.path.join(root, 'data', 'koreaherald_1517_' + str(x) + '.json'), 'r') as f:
      data = json.load(f)
  df = pd.DataFrame(data)
  frames.append(df)

### Put all the articles in one dataframe
all_articles = pd.concat(frames).reset_index(drop=True)
all_articles.columns = ['title', 'author', 'time', 'description', 'body', 'section']

### Change the index to time
all_articles.set_index('time', drop=True, inplace=True)
all_articles = all_articles.sort_index()
all_articles['year'] = pd.DatetimeIndex(all_articles.index).year
all_articles['month'] = pd.DatetimeIndex(all_articles.index).month
all_articles['sorted_id'] = [i for i in range(all_articles.shape[0])]
### Remove 2018
all_articles = all_articles[all_articles['year'] != 2018]

### Preprocess the data
stop_words = set(stopwords.words('english'))

contractions = {
  "aren't": "are not", "can't": "cannot", "couldn't": "could not", "could've": "could have", "didn't": "did not", "doesn't": "does not",
  "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
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
  "who're": "who are", "who's": "who is", "who've": "who have", "who'll": "who will",
  "won't": "will not", "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
  "you're": "you are", "you've": "you have", "wasn't": "was not", "we'll": " will",
  "didn't": "did not", "y'all": "you all", "y'all'd": "you all would", "y'all're": "you all are"
}

def data_cleaning(text):
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', text) # clean url
    text = re.sub(r'#(\w+)', '', text)   # clean hashtags
    text = re.sub(r'@(\w+)', '', text)   # clean @s
    text = re.sub(r'<[^>]+>', '', text)  # clean tags
    text = re.sub(r'\d+', '', text)      # clean digits
    text = re.sub(r'’', '\'', text)      # replace ’ with '
    text = re.sub(r's\'', '', text)      # clean s'
    text = re.sub(r'[£₹$€₩]', ' ', text) # clean currency symbols
    text = re.sub(r'[δ∫βωδσ∈∆≡απθ+*-=°^×√÷]', ' ', text) # clean math symbols
    text = re.sub(r'[/(),!@"“”?.%_&#:;><{}~\[\]|…]', ' ', text)   # clean punctuation
    text = [contractions[word] if word in contractions else word for word in text.split()]  # change contractions to full forms
    #text = [word if not in stop_words for word in text]
    temp = []
    for word in text:
          if word not in stop_words:
                temp.append(word)
    text = temp
    text = [PorterStemmer().stem(word) for word in text] # stem
    text = [WordNetLemmatizer().lemmatize(word) for word in text] # lemmatize
    text = " ".join(text)
    text = re.sub(r'\'s', '', text)      # clean 's
    text = re.sub(r'\'', '', text)       # clean '
    text = re.sub(r'yonhap','', text)    # clean yonhap
    return text

def cloud(text, title, size = (10, 7)):
    words_list = text.unique().tolist()
    words = ' '.join(words_list)
    wordcloud = WordCloud(width = 800, height = 400, collocations = False).generate(words)

    # Output Visualization
    plt.figure(figsize = size, dpi = 80, facecolor = "k", edgecolor = "k")
    plt.imshow(wordcloud,interpolation = "bilinear")
    plt.axis("off")
    plt.title(title, fontsize=25,color = "w")
    plt.tight_layout(pad = 0)
    plt.show()

def clean_title(text):
  return re.sub("[\(\[].*?[\)\]]", "", text)

### Clean the title part of each article and add it as a new column
all_articles['cleaned_title'] = all_articles['title'].apply(data_cleaning)

### Clean the title in place.
all_articles['title'] = all_articles['title'].apply(clean_title)

### Clean the body part of each article and add it as a new column
all_articles['cleaned_body'] = all_articles['body'].apply(data_cleaning)

"""K-Means clustering"""

documents = all_articles['cleaned_body'].values.astype("U")
vectorizer = TfidfVectorizer(stop_words = 'english')
features = vectorizer.fit_transform(documents)

k = 50
model = KMeans(n_clusters=k, init='k-means++', max_iter=100)
model.fit(features)

all_articles['cluster'] = model.labels_

clusters = all_articles.groupby('cluster')

"""Word Cloud per cluster."""

stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=10,
        max_font_size=20,
        scale=3,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


def query(selected_title, per_word = True):
  selected_title = selected_title.lower().split()
  cluster_query = {}
  for cluster_num in range(k):
    doc = clusters.get_group(cluster_num)['title'].values.astype("U")
    if per_word:
      count = sum(collections.Counter(' '.join(doc).split())[word] for word in selected_title)
    cluster_query[cluster_num] = count
  # plt.bar(cluster_query.keys(), cluster_query.values(), )
  # plt.show()
  return {v: k for k, v in cluster_query.items()}[max(cluster_query.values())], cluster_query


topics_year = get_topics_year()

topics = [' '.join(topic) for topic in topics_year[2015]['keyword'][1:11]]

for topic in topics:
  query(topic)




corenlp_dir = './corenlp'
stanza.install_corenlp(dir=corenlp_dir)
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

os.environ["CORENLP_HOME"] = corenlp_dir

# Construct a CoreNLPClient with some basic annotators, a memory allocation of 4GB, and port number 9001
client = CoreNLPClient(
    annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner'],
    memory='4G',
    endpoint='http://localhost:9001',
    be_quiet=True)

client.start()
time.sleep(10)


for issue in topics:
  cluster_num = query(issue)[0]
  cluster_center = model.cluster_centers_[cluster_num]
  cluster_features_ind = [ind for ind in clusters.get_group(cluster_num)['sorted_id']]
  euclidean_dist = {}

  for i in cluster_features_ind:
    f = features[i].toarray()
    euclidean_dist[np.linalg.norm(f - cluster_center)] = i

  top_10 = [euclidean_dist[dist] for dist in sorted(list(euclidean_dist.keys())[:10], reverse = True)]
  top_10.sort()
  top_10_events = all_articles.loc[all_articles.sorted_id.isin(top_10)]

  texts = top_10_events['body']
  titles =  top_10_events['title']
  ind = 0

  print('[ Issue ]', end = '\n\n')

  print(issue, end = '\n\n')

  print('[ On-Issue Events ]', '\n\n')

  print(' -> '.join (titles), end = '\n\n')

  print('[ Detailed Information (per event) ]', end = '\n\n')

  for text in texts:
    print('Event:', titles[ind], end='\n\n')

    person = []
    place = []
    organization = []
    document = client.annotate(text)

    for i, sent in enumerate(document.sentence):
      words = []
      ners = []
      for t in sent.token:
        words.append(t.word)
        ners.append(t.ner)

      while i < len(ners):
        if ners[i] == 'PERSON':
          name = []
          while i < len(ners) and ners[i] == 'PERSON':
            name.append(words[i])
            i += 1
          person.append(' '.join(name))
        elif ners[i] in ['COUNTRY', 'LOC']:
          name = []
          while i < len(ners) and ners[i] in ['COUNTRY', 'LOC']:
            name.append(words[i])
            i += 1
          place.append(' '.join(name))
        elif 'ORG' in t.ner and t.word != 'Yonhap':
          organization.append(t.word)
        i += 1

    doc = nlp(text)
    for sent in doc.sentences:
      for ent in sent.ents:
        if ent.type == 'ORG':
          organization.append(ent.text)
        elif ent.type == 'LOC':
          place.append(ent.text)
    person = list(set(person))
    place = list(set(place))
    organization = list(set(organization))
    print("\t Person: ", ', '.join(list(set(person))))
    print("\t place: ", ', '.join(list(set(place))))
    print("\t organization: ", ', '.join(list(set(organization))), end = '\n\n')

    ind += 1


