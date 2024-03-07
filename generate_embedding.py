from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import pymongo
from nltk.corpus import stopwords
import re

client = pymongo.MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.ThesisProject
collection = db.Thesis_Collection

# nltk.download('punkt')
# nltk.download('stopwords')

stopwords = set(stopwords.words('english'))  

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = ' '.join([word for word in text.split() if word not in stopwords])
    return tokens

print(preprocess_text(collection.find_one()['abstract'][:200]))
