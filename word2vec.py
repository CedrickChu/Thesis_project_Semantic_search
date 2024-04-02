import pymongo
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from typing import List

# Assuming you've loaded your Word2Vec model somehow
# Example:
from gensim.models import Word2Vec
load_model = Word2Vec.load("models/word2vec_model.gensim")

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.ThesisProject
collection = db.Thesis_Collection

def generate_embedding(text: str, model) -> List[float]:
    words = text.split()
    embedding_dim = model.vector_size
    embeddings = np.zeros((len(words), embedding_dim))
    for i, word in enumerate(words):
        if word in model.wv:
            embeddings[i] = model.wv[word]
    if embeddings.shape[0] > 0:
        embedding = np.mean(embeddings, axis=0)
    else:
        embedding = np.zeros(embedding_dim)
    return embedding.tolist()

st.title("Thesis Search")

query = st.text_input("Enter your search query:")

if st.button("Search"):
    query_embedding = generate_embedding(query, load_model)
    results = collection.aggregate([
        {"$search": {
            "index": "index_mapping",  
            "compound": {
                "must": [{
                    "text": {
                        "query": query,
                        "path": "abstract"
                    }
                }]
            }
        }},
        {"$limit": 15}
    ])
    for document in results:
        abstract_embedding = generate_embedding(document["abstract"], load_model)
        similarity_score = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(abstract_embedding).reshape(1, -1))[0][0]
        st.write(f'Thesis Name: {document["title"]},\nAbstract: {document["abstract"]} \n Similarity Score: {similarity_score}\n')
