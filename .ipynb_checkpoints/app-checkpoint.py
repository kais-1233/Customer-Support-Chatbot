# app.py
import streamlit as st
import pickle
import random
from textblob import TextBlob
import json

# ----------------------------
# Load trained model, vectorizer, label encoder
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
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# ----------------------------
# Helper: Spell correction
# ----------------------------
def correct_spelling(text):
    return str(TextBlob(text).correct())

# ----------------------------
# Get bot response
# ----------------------------
def get_response(user_input):
    # Step 1: Correct spelling
    corrected = correct_spelling(user_input)
    
    # Step 2: Vectorize input
    input_vect = vectorizer.transform([corrected])
    
    try:
        # Step 3: Predict probabilities
        probs = model.predict_proba(input_vect)[0]
        pred_index = probs.argmax()
        predicted_tag = le.inverse_transform([pred_index])[0]
        confidence = probs[pred_index]

        # Debug info (optional)
        # st.write(f"DEBUG: Predicted tag: {predicted_tag}, Confidence: {confidence}")

    except Exception as e:
        return "**Bot:** Sorry, I can't understand that. Can you rephrase?"

    # Step 4: Confidence threshold (adjust as needed)
    threshold = 0.2  # Lowered threshold for small dataset
    if confidence < threshold:
        return "**Bot:** Sorry, I can't understand that. Can you rephrase?"

    # Step 5: Return response from predicted intent
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            resp = random.choice(intent.get("responses", []))
            if resp:
                return f"**Bot:** {resp}"
            break

    # Step 6: Extra fallback
    return "**Bot:** Sorry, I can't understand that. Can you rephrase?"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Flipkart Customer Support Bot", page_icon="🤖")
st.title("Flipkart Customer Support Bot")
st.markdown("Type your query below. You can ask about **orders, returns, or refunds.**")

user_input = st.text_input(
    "You:",
    placeholder="Ask me anything about orders, returns, or refunds..."
)

if st.button("Send") and user_input:
    response = get_response(user_input)
    st.markdown(response)

if st.button("Clear Chat"):
    st.experimental_rerun()
