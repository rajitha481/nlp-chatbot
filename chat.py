import json
import pickle
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load intents
with open("data/intents.json", "r") as file:
    intents = json.load(file)

# Load trained data
with open("words.pkl", "rb") as file:
    words = pickle.load(file)

with open("classes.pkl", "rb") as file:
    classes = pickle.load(file)

# Load trained model
model = load_model("chatbot_model.h5")


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
        for i, w in enumerate(words):
            if w == word:
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

    results.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            "intent": classes[r[0]],
            "probability": str(r[1])
        }
        for r in results
    ]


def get_response(intents_list, intents_json):

    if not intents_list:
        return "Sorry, I didn't understand that. Please try again."

    tag = intents_list[0]["intent"]

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."


print("Chatbot is ready!")
print("Type 'quit' to exit.\n")

while True:

    message = input("You: ")

    if message.lower() == "quit":
        print("Bot: Goodbye!")
        break

    ints = predict_class(message)
    response = get_response(ints, intents)

    print("Bot:", response)