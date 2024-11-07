# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('imdb_simple_rnn.h5')

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.markdown(
    """
    <style>
    .stButton > button {
        color: white;
        background-color: #FF5733;
    }
    .stTextArea {
        background-color: #f5f5f5;
        border: 1px solid #ccc;
    }
    .stAlert {
        border-radius: 5px;
        padding: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader("About")
st.write("This application uses a Recurrent Neural Network (RNN) model to classify the sentiment of movie reviews as either positive or negative.")

st.write("### Enter a movie review to classify it:")

# User input
user_input = st.text_area("Movie Review", placeholder="Type your review here...")

# Add a button to trigger the classification
if st.button("Classify Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review before classifying.")
    else:
        # Preprocess the input and predict sentiment
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"
        
        # Display results in an attractive format
        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Prediction Score:** {prediction[0][0]:.2f}")
else:
    st.write("Enter a review and click 'Classify Sentiment' to see the result.")

st.markdown("---")