import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')
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


def preprocess(text, stem_lemmatize=True):
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
    if stem_lemmatize:
        text = [PorterStemmer().stem(word) for word in text]  # stem
        text = [WordNetLemmatizer().lemmatize(word) for word in text]  # lemmatize
    text = " ".join(text)
    text = re.sub(r'\'s', '', text)  # clean 's
    text = re.sub(r'\'', '', text)  # clean '
    text = re.sub(r'yonhap', '', text)  # clean yonhap
    return text
