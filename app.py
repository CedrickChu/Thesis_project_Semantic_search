import pymongo
import requests
import streamlit as st

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.ThesisProject
collection = db.Thesis_Collection


hf_token = "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"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        embedding_url,
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text}
    )

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

    return response.json()

st.title("Thesis Search")

query = st.text_input("Enter your search query:")

if st.button("Search"):
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
        st.write(f'Thesis Name: {document["title"]},\nAbstract: {document["abstract"]}\n')
