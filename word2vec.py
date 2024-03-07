import streamlit as st
import gensim.models as gm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
import pymongo
from sklearn.metrics.pairwise import cosine_similarity


trained_model = gm.Word2Vec.load('word2vec_model.gensim')

stop_words = set(stopwords.words('english'))

client = pymongo.MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.ThesisProject
collection = db.Thesis_Collection

def preprocess_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def get_query_vector(word2vec_model, query_tokens):
    valid_tokens = [word for word in query_tokens if word in word2vec_model.wv]
    if not valid_tokens:
        return np.zeros(word2vec_model.vector_size)
    query_vector = np.mean([word2vec_model.wv[word] for word in valid_tokens], axis=0)
    return query_vector


def calculate_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def find_most_relevant_docs(query_text, trained_model):
    query_tokens = preprocess_text(query_text)
    query_vector = get_query_vector(trained_model, query_tokens)
    most_relevant_docs = []

    for doc in collection.find():
        doc_text = doc['abstract']
        title = doc['title']
        doc_tokens = preprocess_text(doc_text)
        
        doc_vector = get_query_vector(trained_model, doc_tokens)

        if np.all(doc_vector == 0) or np.all(query_vector == 0):
            continue

        similarity = cosine_similarity(query_vector.reshape(1, -1), doc_vector.reshape(1, -1))[0, 0]

        if not np.isnan(similarity):
            most_relevant_docs.append({'_id': doc['_id'], 'title': title, 'similarity': similarity, 'abstract': doc_text})

    most_relevant_docs = sorted(most_relevant_docs, key=lambda x: x['similarity'], reverse=True)
    return most_relevant_docs

st.title('Document Relevance Query')

query_text = st.text_input("Enter your query:", "")

if query_text:
    most_relevant_docs = find_most_relevant_docs(query_text, trained_model)
    
    num_abstracts_to_print = min(15, len(most_relevant_docs))
    for i in range(num_abstracts_to_print):
        current_doc = most_relevant_docs[i]
        st.write(f"Document ID: {current_doc['_id']}")
        st.write(f"Title: {current_doc['title']}")
        st.write(f"Similarity: {current_doc['similarity']}")
        st.write(f"Abstract:\n{current_doc['abstract']}")
        st.write("\n" + "="*50 + "\n")