import pymongo
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from typing import List

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = ' '.join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])
    return tokens

client = pymongo.MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.ThesisProject
collection = db.Thesis_Collection

def custom_vectorizer(corpus):
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS))
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def generate_embedding(text: str, vectorizer: TfidfVectorizer) -> List[float]:
    preprocessed_text = preprocess_text(text)
    embedding_matrix = vectorizer.transform([preprocessed_text])
    embedding = embedding_matrix.toarray().flatten().tolist()
    return embedding

query = st.text_input("Enter your search query:")

if st.button("Search"):
    corpus = [doc["abstract"] for doc in collection.find({}, {"abstract": 1})]
    X, vectorizer = custom_vectorizer(corpus)

    query_embedding = generate_embedding(query, vectorizer)

    results = collection.find({}, {"title": 1, "abstract": 1})
    for document in results:
        doc_embedding = generate_embedding(document["abstract"], vectorizer)
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        st.write(f'Thesis Name: {document["title"]},\nAbstract: {document["abstract"]}\nCosine Similarity: {similarity}\n')
