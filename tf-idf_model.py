import pymongo
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from typing import List
from gensim.models import Word2Vec
import numpy as np


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = ' '.join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])
    return tokens

client = pymongo.MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.ThesisProject
collection = db.Thesis_Collection

# Function to train a Word2Vec model
def train_word2vec_model(corpus):
    tokenized_corpus = [preprocess_text(text).split() for text in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    return model

def generate_embedding(text: str) -> List[float]:
    preprocessed_text = preprocess_text(text)
    model = train_word2vec_model([preprocessed_text])
    
    # Calculate the mean of word vectors in the document
    word_vectors = [model.wv[word] for word in preprocessed_text.split() if word in model.wv]
    if not word_vectors:
        # If no words in the document are in the vocabulary, return zeros
        return [0.0] * model.vector_size
    
    embedding = np.mean(word_vectors, axis=0)
    return embedding.tolist()


query = "Machine Learning"

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding_hf",
    "numCandidates": 100,
    "limit": 4,
    "index": "abstract_embedding",
      }}
])

for document in results:
    print(f'Thesis Name: {document["title"]},\Thesis Abstract: {document["plot"]}\n')