# app.py
import streamlit as st
import pickle
import random
from textblob import TextBlob

# ----------------------------
# Load saved model, vectorizer, label encoder
# ----------------------------
with open("flipkart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("flipkart_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("flipkart_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ----------------------------
# Load intents JSON
# ----------------------------
import json
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# ----------------------------
# Streamlit UI setup
# ----------------------------
st.set_page_config(page_title="Flipkart Customer Support Bot", page_icon="")
st.title("Flipkart Customer Support Bot")
st.markdown("Type your query below. The bot will respond based on trained intents.")

# ----------------------------
# Helper function: text preprocessing / spell correction
# ----------------------------
def correct_text(text):
    return str(TextBlob(text).correct())

# ----------------------------
# Chatbot response function
# ----------------------------
def get_response(user_input):
    corrected = correct_text(user_input)
    
    input_vect = vectorizer.transform([corrected])
    probs = model.predict_proba(input_vect)
    pred_index = probs.argmax()
    predicted_tag = le.inverse_transform([pred_index])[0]
    
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            resp = random.choice(intent.get("responses", ["Sorry, I couldn't find an answer."]))
            return f"**Bot:** {resp}"

# ----------------------------
# Streamlit user input
# ----------------------------

user_input = st.text_input(
    "You:",
    placeholder="Ask me about order, return,exchange,refund...",
)

if st.button("**Send**") and user_input:
    response = get_response(user_input)
    st.markdown(response)

if st.button("**Clear Chat**"):
    st.experimental_rerun()
