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
st.title("Flipkart Customer Support ChatBot")
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

 



# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --- Chat container with fixed height and scroll ---
chat_container = st.container()
chat_height = 400  # height in pixels

# --- Display chat inside scrollable div ---
def display_chat():
    chat_html = "<div style='height:{}px; overflow-y: auto; border:1px solid #ddd; padding:10px;'>".format(chat_height)
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            chat_html += f"<p><b style='color:blue'>{sender}:</b> {msg}</p>"
        else:
            chat_html += f"<p><b style='color:green'>{sender}:</b> {msg}</p>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

# --- Text input ---
user_input = st.text_input(
    "You:",
    placeholder="Ask me about order, return, exchange, refund...",
    key="user_input"
)

# --- Send button ---
def send_message():
    if st.session_state.user_input.strip():
        response = get_response(st.session_state.user_input)
        st.session_state.chat_history.append(("You", st.session_state.user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.session_state.user_input = ""  # clear input

st.button("Send", on_click=send_message)

# --- Clear chat button ---
def clear_chat():
    st.session_state.chat_history = []
    st.session_state.user_input = ""

st.button("Clear Chat", on_click=clear_chat)

# --- Render chat ---
display_chat()
