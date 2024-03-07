from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import pymongo
from nltk.corpus import stopwords
import re
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.ThesisProject
collection = db.Thesis_Collection
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens
def sentence_tokenization(text: str) -> List[List[str]]:
    tokenized_sentences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    return tokenized_sentences
def create_corpus(text: str) -> List[List[str]]:
    word_tokens = preprocess_text(text)
    sentence_tokens = sentence_tokenization(text)
    corpus = [word_tokens] + sentence_tokens
    return corpus
def get_query_vector(word2vec_model, query_tokens):
    valid_tokens = [word for word in query_tokens if word in word2vec_model.wv]

    if not valid_tokens:
        return np.zeros(word2vec_model.vector_size)

    query_vector = np.mean([word2vec_model.wv[word] for word in valid_tokens], axis=0)
    return query_vector
def train_word2vec_model(corpus):
    VECTOR_SIZE = 100
    WINDOW = 10
    MIN_COUNT = 1
    SG = 0
    model = Word2Vec(
        sentences=corpus,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG
    )
    model.save('word2vec_model.gensim')
    return model

all_abstracts = [doc['abstract'] for doc in collection.find()]
corpus = create_corpus(" ".join(all_abstracts))
print(corpus)