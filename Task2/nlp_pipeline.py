# nlp_pipeline.py

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, ngrams
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
    
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    
    return lemmatized_tokens

def get_word_freq(tokens):
    return Counter(tokens).most_common(20)

def get_pos_tags(tokens):
    return pos_tag(tokens)

def get_ngrams(tokens, n=2):
    return Counter(ngrams(tokens, n)).most_common(10)

def tokens_to_dataframe(tokens):
    return pd.DataFrame(tokens, columns=['Token'])

def get_top_pos(pos_tags, top_n=10):
    pos_only = [tag for word, tag in pos_tags]
    return Counter(pos_only).most_common(top_n)
