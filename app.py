import streamlit as st
import json
import pickle
import random
import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Rajitha's NLP Chatbot",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------------
# LOAD CHATBOT
# -----------------------------------

lemmatizer = WordNetLemmatizer()


@st.cache_resource
def load_chatbot():

    with open("data/intents.json", "r") as file:
        intents = json.load(file)

    with open("words.pkl", "rb") as file:
        words = pickle.load(file)

    with open("classes.pkl", "rb") as file:
        classes = pickle.load(file)

    model = load_model("chatbot_model.h5")

    return intents, words, classes, model


intents, words, classes, model = load_chatbot()


# -----------------------------------
# NLP FUNCTIONS
# -----------------------------------

def clean_up_sentence(sentence):

    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [
        lemmatizer.lemmatize(word.lower())
        for word in sentence_words
    ]

    return sentence_words


def bag_of_words(sentence):

    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)

    for word in sentence_words:

        for i, word_in_words in enumerate(words):

            if word_in_words == word:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):

    bow = bag_of_words(sentence)

    result = model.predict(
        np.array([bow]),
        verbose=0
    )[0]

    ERROR_THRESHOLD = 0.25

    results = [
        [i, probability]
        for i, probability in enumerate(result)
        if probability > ERROR_THRESHOLD
    ]

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "intent": classes[result[0]],
            "probability": str(result[1])
        }
        for result in results
    ]


def get_response(intents_list):

    if not intents_list:
        return "Sorry, I didn't understand that. Please try again."

    tag = intents_list[0]["intent"]

    for intent in intents["intents"]:

        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."


# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    st.title("🤖 Chatbot")

    st.write("### About")

    st.write(
        """
        This chatbot uses:

        • NLP  
        • NLTK  
        • Bag of Words  
        • TensorFlow  
        • Neural Networks  
        • Streamlit
        """
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# -----------------------------------
# MAIN UI
# -----------------------------------

st.title("🤖 AI Chatbot")

st.caption(
    "An NLP-based chatbot built using Python, TensorFlow and Streamlit."
)

st.divider()


# -----------------------------------
# CHAT HISTORY
# -----------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! 👋 I'm your AI chatbot. How can I help you?"
        }
    ]


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# -----------------------------------
# USER INPUT
# -----------------------------------

user_input = st.chat_input("Type your message here...")


if user_input:

    # Display user message
    with st.chat_message("user"):

        st.write(user_input)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # Get chatbot response
    with st.spinner("🤖 Thinking..."):

        intents_list = predict_class(user_input)

        response = get_response(intents_list)


    # Display bot response
    with st.chat_message("assistant"):

        st.write(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )