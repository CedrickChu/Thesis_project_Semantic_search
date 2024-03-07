import requests
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from typing import List
import re

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return tokens

client = pymongo.MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.ThesisProject
collection = db.Thesis_Collection

def custom_vectorizer(corpus):
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS))
    X = vectorizer.fit_transform(corpus)
    return X

def generate_embedding(text: str) -> List[float]:
    preprocessed_text = preprocess_text(text)
    embedding_matrix = custom_vectorizer([preprocessed_text])
    embedding = embedding_matrix.toarray().flatten().tolist()
    return embedding

for doc in collection.find({'abstract_embedding2': {"$exists": True}}).limit(10):
    abstract_embedding = generate_embedding(doc['abstract'])
    collection.update_one({'_id': doc['_id']}, {"$set": {'abstract_embedding2': abstract_embedding}})
