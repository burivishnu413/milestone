import cohere
import streamlit as st
import pandas as pd
import speech_recognition as sr
import pyaudio
from elasticsearch import Elasticsearch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import os

# Initialize Cohere client
COHERE_API_KEY="Pk7PaQxvrfxAIi3I2QqHiGwalCKlwunTFKCqDj1F"
co = cohere.Client(COHERE_API_KEY)

# Initialize Elasticsearch client with error handling
try:
    es = Elasticsearch("http://localhost:9200")
    index_name = "call_data"
    es.ping()
except Exception as e:
    st.error("Elasticsearch connection failed. Check if the server is running.")
    es = None

# Load product data with error handling
product_data_path = "C:/Users/vishn/OneDrive/Desktop/milestone/amazon.csv"
try:
    product_data = pd.read_csv(product_data_path).fillna('')
except FileNotFoundError:
    st.error("Product data file not found. Please check the path.")
    product_data = pd.DataFrame()

# Load Hugging Face models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label']

def summarize_text(text):
    summary = summarization_pipeline(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def retrieve_context(query, top_k=3):
    if es:
        response = es.search(index=index_name, body={
            "query": {"match": {"content": query}},
            "size": top_k
        })
        return [hit["_source"]["content"] for hit in response["hits"]["hits"]]
    return []

def main():
    st.title("📞 Real-Time Speech Analysis & Product Recommendations")
    st.markdown("Analyze call transcripts and get insights in real time using Hugging Face models.")

    user_input = st.text_area("Enter call transcript or speak now:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyze Sentiment"):
            sentiment = analyze_sentiment(user_input)
            st.success(f"Sentiment: **{sentiment}**")

    with col2:
        if st.button("📄 Summarize Text"):
            summary = summarize_text(user_input)
            st.info(f"**Summary:** {summary}")
    
    st.write("---")
    st.subheader("🎙️ Speak into the microphone")
    recognizer = sr.Recognizer()
    
    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            if st.button("🎤 Start Recording"):
                st.info("Listening... Please speak clearly.")
                audio = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio)
                    st.write(f"Recognized Speech: **{text}**")
                    sentiment = analyze_sentiment(text)
                    st.success(f"Sentiment: **{sentiment}**")
                except sr.UnknownValueError:
                    st.error("Sorry, could not understand the audio.")
                except sr.RequestError:
                    st.error("Could not request results; please check your connection.")
    except Exception as e:
        st.error("Microphone not found or not accessible. Please check your audio settings.")

if __name__ == "__main__":
    main()
