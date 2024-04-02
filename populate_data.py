from datetime import datetime
import random
from faker import Faker
from pymongo import MongoClient

fake = Faker()

def generate_fake_data(num_entries):
    data = []
    for _ in range(num_entries):
        title = fake.sentence(nb_words=6, variable_nb_words=True, ext_word_list=None)
        author = fake.name()
        date_published = datetime.combine(fake.date_this_decade(), datetime.min.time())  # Convert date to datetime
        keywords = ', '.join([fake.word() for _ in range(random.randint(3, 6))])  # Join keywords into a single string
        abstract = fake.paragraph(nb_sentences=4, variable_nb_sentences=True, ext_word_list=None)
        
        entry = {
            'title': title,
            'author': author,
            'date_published': date_published,
            'keywords': keywords,
            'abstract': abstract
        }
        data.append(entry)
    return data

def insert_data_into_mongodb(data, collection):
    result = collection.insert_many(data)
    print(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")

if __name__ == "__main__":
    num_entries = 20
    data = generate_fake_data(num_entries)
    
    client = MongoClient("mongodb+srv://cedrickchu123:lzuaguRde81CZVuD@cluster0.75dzsfe.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client['ThesisProject']
    collection = db['Thesis_Collection']
    
    insert_data_into_mongodb(data, collection)
