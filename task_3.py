import pandas as pd
import numpy as np
import re, os, json
import nltk
import spacy
import pathlib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords
from bertopic import BERTopic
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from easydict import EasyDict as edict
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from umap import UMAP
from task_1 import get_topics_year
nltk.download('stopwords')



class Preprocess():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.contractions = {
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

    def merge_data(self):
        frames = []
        for x in range(8):
            with open(os.path.join(pathlib.Path.cwd(), 'data', 'koreaherald_1517_' + str(x) + '.json'), 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            frames.append(df)

        ### Put all the articles in one dataframe
        merged_articles = pd.concat(frames).reset_index(drop=True)
        merged_articles.columns = ['title', 'author', 'time', 'description', 'body', 'section']
        merged_articles = merged_articles.sort_values(by=['time'], ignore_index=True)     ## Sort by time
        merged_articles['year'] = pd.DatetimeIndex(merged_articles['time']).year    ## Insert years
        merged_articles['month'] = pd.DatetimeIndex(merged_articles['time']).month    ## Insert months
        merged_articles = merged_articles[merged_articles['year'] != 2018]     ## Remove 2018
        merged_articles['section'].replace("", "Unlabeled", inplace=True)       ## Fill unlabeled sections
        return merged_articles

    def data_cleaning(self, text):
        text = text.lower()
        text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', text) # clean url
        text = re.sub(r'#(\w+)', '', text)   # clean hashtags
        text = re.sub(r'@(\w+)', '', text)   # clean @s
        text = re.sub(r'<[^>]+>', '', text)  # clean tags
        text = re.sub(r'\w*[0-9]\w*', '', text)   # clean any number word combination
        text = re.sub(r'[’‘]', '\'', text)   # replace ’‘ with '
        text = re.sub(r's\'', '', text)      # clean s'
        text = re.sub(r'[£₹$€₩]', ' ', text) # clean currency symbols
        text = re.sub(r'[δ∫βωδσ∈∆≡απθ+*-=°^×√÷]', '', text) # clean math symbols
        text = re.sub(r'[/(),!@"“”?%_&#:;><{}~\[\]|…]', '', text)   # clean punctuation 
        text = text.split()
        
        temp = []
        for word in text:
            if word=="u.s.":
                word="us"
            if word=="graphic" or word=="news" or word=="said":
                continue
            temp.append(word)
        text = " ".join(temp)
        text = re.sub(r'[.]', '', text)
        text = [self.contractions[word] if word in self.contractions else word for word in text.split()]  # change contractions to full forms
        
        temp = []
        for word in text:
            if word=="n":
                word="north"
            elif word=="s":
                word="south"
            elif word=="k":
                word="korea"
            elif word=="nk":
                word="north korea"
            if word not in self.stop_words:
                temp.append(word)
        text = temp
        text = " ".join(text)
        text = re.sub(r'\'s', '', text)      # clean 's
        text = re.sub(r'\'', '', text)       # clean '
        text = re.sub(r'yonhap','', text)    # clean yonhap
        return text

    def cloud(self, text, title, size = (10, 7)):
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

    def print_aritcle(self, data, art_number):
        if art_number >= data.shape[0]:
            return "Article doesn't exist"
        print("Article BEFORE preprocessing\n", data.iloc[art_number, data.columns.get_loc("title")])
        print(data.iloc[art_number, data.columns.get_loc("body")])
        print("\nArticle AFTER preprocessing\n", data.iloc[art_number, data.columns.get_loc("cleaned_title")])
        print(data.iloc[art_number, data.columns.get_loc("cleaned_body")])

    def go_clean(self):
        merged_data = self.merge_data()
        merged_data['cleaned_title'] = merged_data['title'].apply(self.data_cleaning)      ## Clean the title part of each article and add it as a new column
        merged_data['cleaned_body'] = merged_data['body'].apply(self.data_cleaning)      ## Clean the body part of each article and add it as a new column
        return merged_data



class TopicModel():
    def __init__(self, all, years=[2015, 2016, 2017]):
        self.topic_model = BERTopic(embedding_model='all-mpnet-base-v2') #all-mpnet-base-v2 #all-MiniLM-L6-v2
        self.all = all
        self.years = years

    def topic_top(self, top=10):
        topics, _ = self.topic_model.fit_transform(self.all[self.all['year'].isin(self.years)]['cleaned_body'])
        my_dict = {}
        
        for i in range(1, top + 1):
            my_dict[i] = self.topic_model.get_topic(i-1)
        return my_dict
  
    def topic_visualize(self):
        return self.topic_model.visualize_barchart()



args = edict()
args.reduceTo = 5   #vector dimension after reduction
args.clusters = 10  # number of clusters
args.max_iter = 500   # maximum number of iterations for clustering
args.docs = 10    #once moving to the negihbouring cluster pick every args.docs(th) doc for relation extraction
args.how_many_docs = 20   #how many docs within a cluster to look at to find the closest clusters
args.how_many_related = 3 #how many related issues to print per issue
args.cluster_freq = 0 #cluster freq is in desceding order, so most frequent will have index 0. Change this number to start from other clusters
args.neighbouring_cluster = 1 # 1(immediate neighbour) - args.clusters-1(furthest)
args.top_issues = 5 #how many top issues do you want to consider


class Wrapper():
    def __init__(self):
        pass

    def cosine_similarity(self, A, B):
        np.seterr(divide='ignore', invalid='ignore')
        B_T = B.T
        product = np.matmul(A, B_T)

        normA = np.linalg.norm(A, axis=1)
        normA = normA.reshape(normA.size, 1)

        normB_T = np.linalg.norm(B_T, axis=0)
        normB_T = normB_T.reshape(1, normB_T.size)
        product_norms = np.matmul(normA, normB_T)
        similarity = np.subtract(1, np.divide(product,product_norms))
        return similarity

    def elbow_method(self, cluster, max_):
        number_clusters = range(1, max_)  # Range of possible clusters that can be generated
        kmeans = [KMeans(n_clusters=i, max_iter = args.max_iter) for i in number_clusters] # Getting no. of clusters 

        score = [kmeans[i].fit(cluster).score(cluster) for i in range(len(kmeans))] # Getting score corresponding to each cluster.
        score = [i*-1 for i in score] # Getting list of positive scores.
        
        plt.plot(number_clusters, score)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Elbow Method')
        plt.show()

    def closest_centroid(self, centroids, center, p=2):
        if p==1:
            L1_distance = np.sum(np.abs(centroids - centroids[center, :]), axis = 1)
            min_dist_index = np.argsort(L1_distance)
        elif p==2:
            L2_distance = np.sum(np.square(centroids - centroids[center, :]), axis = 1)
            min_dist_index = np.argsort(L2_distance)
        return min_dist_index[1:]

    def cluster(self, data, top_issues, reduce=True, method="umap", vectorizer="tfidf"):
        if vectorizer == "tfidf":
            model = TfidfVectorizer()
            X = model.fit(data['cleaned_body'])
            print(f"Embedding article body {'.'*10}\n")
            embeddings_body = X.transform(data['cleaned_body'])
            print(f"Embedding article title {'.'*10}\n")
            embeddings_title = X.transform(data['cleaned_title'])
            print(f"Embedding article issues {'.'*10}\n")
            embeddings_issues = X.transform([" ".join(x) for x in top_issues])
        else:
            if vectorizer == "bert":
                model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            elif vectorizer == "roberta":
                model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
            elif vectorizer == "mpnet":
                model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

            print(f"Embedding article body {'.'*10}\n")
            embeddings_body = model.encode(data["cleaned_body"], show_progress_bar=False)
            print(f"Embedding article title {'.'*10}\n")
            embeddings_title = model.encode(data["cleaned_title"], show_progress_bar=False)
            print(f"Embedding article issues {'.'*10}\n")
            embeddings_issues = model.encode(top_issues, show_progress_bar=False)
            
        if reduce == True:
            if method == "pca":
                X_ = TruncatedSVD(n_components=args.reduceTo, random_state=42).fit(embeddings_body)
   
            elif method == "umap":
                X_ = UMAP(n_components=args.reduceTo, min_dist=0, metric='cosine').fit(embeddings_body)
            
            print(f"Reducing each embeddings dimensionality from {embeddings_body.shape[1]} to {args.reduceTo} {'.'*10}\n")
            embeddings_body = X_.transform(embeddings_body)
            embeddings_title = X_.transform(embeddings_title)
            embeddings_issues = X_.transform(embeddings_issues)
    
        kmeans = KMeans(n_clusters = args.clusters, init = 'k-means++', max_iter = args.max_iter, n_init = 1).fit(embeddings_body)
        data['cluster'], centroids = kmeans.labels_, kmeans.cluster_centers_

        x = self.cosine_similarity(embeddings_body, embeddings_issues)
        nlp = spacy.load("en_core_web_lg")
        topic_model = BERTopic()
        print("*"*100)
        print("*"*100)
        print()

        for issue_num in range(args.top_issues): #len(top_issues)
            closest_similarity_indices = np.argsort(x[:, issue_num])
            
            closest_clusters = []   # label of doc clusters of the top most closest docs to the issue 
            for i in range(args.how_many_docs): # take the first x docs that are most closest to the issue and append their cluster label
                closest_clusters.append(data.iloc[closest_similarity_indices[i], data.columns.get_loc("cluster")])
            
            closest_clusters_dict = Counter(closest_clusters)
            labels = sorted(closest_clusters_dict, key=closest_clusters_dict.get, reverse=True)   #sort descending in label frequency
            closest_cluster_labels = self.closest_centroid(centroids, labels[args.cluster_freq], 2)    #list of clusters from closest to furthest of the most abundant cluster in closest_clusters

            index_list = data[data['cluster'] == closest_cluster_labels[args.neighbouring_cluster - 1]].index.values
            embeddings_neighbouring_cluster = embeddings_body[index_list]   #embeddings of only the neighbouring cluster
            x = self.cosine_similarity(embeddings_neighbouring_cluster, embeddings_issues)
            closest_similarity_indices = np.argsort(x[:, 1])

            print('[ISSUE {}]\n'.format(issue_num + 1))
            print(' '.join(set(top_issues[issue_num][:3])).capitalize())
            print('\n\t [Related-Issue Events] \n')

            for i in range(args.how_many_related):
                topics_index = [index_list[index] for index in closest_similarity_indices[i*args.docs : (i*args.docs) + 5]]
                topics_data = all_articles.iloc[topics_index]
                topics_data = pd.concat([topics_data]*20, ignore_index=True)
                topics_data = topics_data.sample(frac=1).reset_index(drop=True)

                if vectorizer == "tfidf":
                    embeddings_topic = X.transform(topics_data['cleaned_body'])
                else:
                    embeddings_topic = model.encode(topics_data['cleaned_body'], show_progress_bar=False)
                
                if reduce == True:
                    embeddings_topic = X_.transform(embeddings_topic)

                topics, _ = topic_model.fit_transform(topics_data['cleaned_body'], embeddings_topic)
                top_first = topic_model.get_topic(0)
                
                if type(top_first) is bool:
                    print('\t\t Title: {}'.format(topics_data.iloc[0, data.columns.get_loc("cleaned_title")]))
                else:
                    print('\t\t {} \n'.format(" ".join([topic[0] for topic in top_first][:3]).capitalize()))

                doc = nlp(data.iloc[index_list[closest_similarity_indices[i*args.docs]], data.columns.get_loc("cleaned_body")])
                locations = []
                organizations = []
                people = []
                country = []

                for entity in doc.ents:
                    if entity.label_ == "LOC":
                        locations.append(entity.text.capitalize())
                    if entity.label_ == "ORG":
                        organizations.append(entity.text.capitalize())
                    if entity.label_ == "PERSON":
                        people.append(entity.text.capitalize())
                    if entity.label_ == "GPE":
                        country.append(entity.text.capitalize())
                
                print("\t\t\t -> PERSON: {}".format(", ".join(set(people))))
                print("\t\t\t -> ORGANIZATION: {}".format(", ".join(set(organizations))))
                print("\t\t\t -> PLACE: {}".format(", ".join(set(locations))))
                print("\t\t\t -> COUNTRY: {} \n".format(", ".join(set(country))))
            
            print("\n")


if __name__ == '__main__':
    process = Preprocess()
    all_articles = process.go_clean()

    topics = get_topics_year() #TopicModel(all_articles, years=[2015, 2016, 2017])
    top_issues = [topics["all"].keyword.iloc[i].split() for i in range(1, 10)]  #[[a[0] for a in x] for x in list(topics.topic_top(10).values())]
    
    go = Wrapper()
    go.cluster(all_articles, top_issues, reduce=True, method="umap", vectorizer="mpnet")