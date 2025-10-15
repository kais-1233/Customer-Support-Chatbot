# Flipkart Customer Service Chatbot

## Project Overview
This is an AI-powered Customer Service Chatbot designed to assist Flipkart users with common queries such as order tracking, returns, cancellations, discounts, payment issues, and delivery timings.  
The chatbot leverages Natural Language Processing (NLP) techniques to understand user inputs and provide accurate responses in real time.

## Features
- Order Tracking: Provides updates on the status of customer orders.
- Return and Cancellation Support: Shares information on return policies and order cancellation steps.
- Discount and Offer Information: Suggests available offers, promo codes, and deals.
- Payment and Refund Queries: Assists users with payment failure issues and refund status.
- Delivery Timings: Displays estimated delivery times for orders.
- Greetings and Farewell Responses: Handles general conversational interactions.

## Technologies Used
- Python 3 – Primary programming language for chatbot development
- NLTK – Text preprocessing tasks (tokenization, stopword removal, lemmatization)
- TextBlob – Spell correction and simple NLP operations during training
- Scikit-learn
  - TF-IDF Vectorizer – Converts user input text into numerical features
  - Logistic Regression – Intent classification
- Pickle – Save and load trained model
- JSON – Store intents, patterns, and responses
- Streamlit – Web interface for live interaction
- Deployment - https://customers-support-chatbot.streamlit.app/

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/kais-1233/Customer-Support-Chatbot.git
