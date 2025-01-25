import pandas as pd
import cohere
import speech_recognition as sr
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gspread
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import streamlit as st

# Initialize Cohere client
COHERE_API_KEY ="cohere api key"
co = cohere.Client(COHERE_API_KEY)

# Initialize Elasticsearch client
es = Elasticsearch("http://localhost:9200")
index_name = "call_data","product_data"

import os

product_data_path = r"C:/Users/vishn/OneDrive/Desktop/milestone/amazon.csv"

if os.path.exists(product_data_path):
    print(f"File found at: {product_data_path}")
    product_data = pd.read_csv(product_data_path).fillna('')
else:
    print(f"File not found at: {product_data_path}")


# Load data
product_data = pd.read_csv(product_data_path).fillna('')
# Verify if the file exists
if st.button("Load Product Data"):
    try:
        product_data = pd.read_csv(product_data_path).fillna('')
        st.success("Product data loaded successfully!")
    except FileNotFoundError:
        st.error("File not found at the specified path.")



# Google Sheets authentication
import gspread
from google.oauth2.service_account import Credentials

def authenticate_google_sheets():
    try:
        # Path to your downloaded JSON key file
        json_file_path = r"C:/Users/vishn/Downloads/root-opus-421017-9bfaf4781ccc.json"
        
        # Load credentials from the JSON key file
        creds = Credentials.from_service_account_file(json_file_path, scopes=["https://www.googleapis.com/auth/spreadsheets"])

        # Authorize the client with credentials
        client = gspread.authorize(creds)

        # Google Sheet ID from your link
        sheet_id = "1oGuE7Ii9hqBsRiEfd9JMowuXvVE7PkorDR0au834fmY"
        
        # Open the Google Sheet using the ID
        sheet = client.open_by_key(sheet_id).sheet1  # Access the first sheet
        
        return sheet
    except Exception as e:
        print(f"Error in Google Sheets authentication: {e}")
        return None


def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] > 0.05:
        return "Positive"
    elif sentiment_scores['compound'] < -0.05:
        return "Negative"
    else:
        return "Neutral"



# Index documents into Elasticsearch
def index_documents(documents, index_name="call_data"):
    for i, doc in enumerate(documents):
        es.index(index=index_name, id=i, document={"content": doc})

def retrieve_context(query, top_k=3):
    response = es.search(index=index_name, body={
        "query": {"match": {"content": query}},
        "size": top_k
    })
    return [hit["_source"]["content"] for hit in response["hits"]["hits"]]

def generate_call_summary(call_script, context):
    combined_input = call_script + "\n\nRelevant Context:\n" + "\n".join(context)
    response = co.generate(
        model="command-xlarge-nightly",
        prompt=f"Call Script:\n{combined_input}\n\nGenerate a concise summary:",
        max_tokens=300, temperature=0.5
    )
    return response.generations[0].text.strip()
    
def handle_objection(query):
    response = co.generate(
        model="command-xlarge",
        prompt=f"Handle the following objection: {query}",
        max_tokens=100, temperature=0.7
    )
    return response.generations[0].text.strip()

    

def main():
    st.title("Real-Time Speech Sentiment Analysis and Product Recommendations")
    
    user_input = st.text_area("Enter call transcript or speak now:")
    if st.button("Analyze Sentiment"):
        sentiment = analyze_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")

    if st.button("Generate Call Summary"):
        context = retrieve_context(user_input)
        summary = generate_call_summary(user_input, context)
        st.write(f"Summary: {summary}")

    if st.button("Handle Objection"):
        response = handle_objection(user_input)
        st.write(f"Objection Response: {response}")
    
    st.write("---")
    st.write("Speak into the microphone")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    if st.button("Start Recording"):
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                st.write(f"Recognized Speech: {text}")
                sentiment = analyze_sentiment(text)
                st.write(f"Sentiment: {sentiment}")
            except Exception as e:
                st.write(f"Error: {e}")
# Elasticsearch product recommendation search
def generate_embedding(text):
    try:
        response = co.embed(texts=[text], model="embed-english-v2.0")
        return response.embeddings[0]
    except Exception as e:
        print(f"Error generating embedding for text: {text}\n{e}")
        return None

# Define index creation function with proper mapping
def create_index_with_mapping(index_name, embedding_dim=768):
    try:
        # Delete index if it exists
        es.indices.delete(index=index_name, ignore=[400, 404])
        
        # Define mapping for the index
        mapping = {
            "mappings": {
                "properties": {
                    "name": {"type": "text"},
                    "price": {"type": "float"},
                    "category": {"type": "text"},
                    "description": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": embedding_dim}
                }
            }
        }
        
        # Create index with the defined mapping
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Error creating index: {e}")

# Index product data into Elasticsearch
def index_products_with_embeddings(product_data, index_name="call_data"):
    try:
        for _, row in product_data.iterrows():
            product_description = f"{row['name']} {row['description']}"
            embedding = generate_embedding(product_description)
            
            if embedding is not None:
                document = {
                    "name": row["name"],
                    "price": row["price"],
                    "category": row["category"],
                    "description": row["description"],
                    "embedding": embedding
                }
                es.index(index=index_name, document=document)
        print("Products indexed successfully.")
    except Exception as e:
        print(f"Error indexing products: {e}")

# Update search function for product recommendations
def search_product(query):
    try:
        query_embedding = generate_embedding(query)
        if query_embedding is None:
            print("Failed to generate embedding for query.")
            return

        # Search for the top product
        search_query = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
        }

        response = es.search(index=index_name, body=search_query)
        hits = response['hits']['hits']

        if hits:
            top_product = hits[0]["_source"]
            print(f"Top Product Recommendation:\nName: {top_product['name']}, Price: {top_product['price']}, Category: {top_product['category']}, Description: {top_product['description']}")

            category = top_product['category']
            similar_products_query = {
                "size": 4,
                "query": {
                    "bool": {
                        "must": [{"match": {"category": category}}],
                        "must_not": [{"term": {"name": top_product['name']}}]  # Exclude the top product
                    }
                }
            }
            similar_products_response = es.search(index=index_name, body=similar_products_query)
            similar_hits = similar_products_response['hits']['hits']

            print(f"Similar products in the {category} category:")
            for hit in similar_hits:
                source = hit["_source"]
                print(f"Name: {source['name']}, Price: {source['price']}, Description: {source['description']}")
        else:
            print("No matching products found.")

        objection_response = handle_objection(query)
        print(f"Objection Response: {objection_response}")

    except Exception as e:
        print(f"Error during search: {e}")

def real_time_analysis():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Say 'stop' to stop the process.")
    try:
        while True:
            with mic as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                print("Recognizing...")
                text = recognizer.recognize_google(audio)
                print(f"Recognized Text: {text}")

                if 'stop' in text.lower():
                    print("Stopping real-time analysis...")
                    break

                # Sentiment analysis
                sentiment = analyze_sentiment(text)
                print(f"Sentiment: {sentiment}")

                # Context retrieval
                context = retrieve_context(text)

                # Call summary
                result = generate_call_summary(text, context)
                print("Key Points:")
                for point in result["Key Points"]:
                    print(f"- {point}")
                print("\nSummary:")
                print(result["Summary"])

                # Objection handling
                objection_response = handle_objection(text)
                print(f"Objection Response: {objection_response}")

                # Product recommendation
                print("Product Recommendation:")
                search_product(text)
            except sr.UnknownValueError:
                print("Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                print(f"Error with the Speech Recognition service: {e}")
            except Exception as e:
                print(f"Error during processing: {e}")

    except Exception as e:
        print(f"Error in real-time analysis: {e}")

if __name__ == "__main__":
    main()

