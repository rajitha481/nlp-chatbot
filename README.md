# 🤖 NLP Chatbot

An intelligent NLP-based chatbot built using **Python, TensorFlow, NLTK, and Streamlit**. The chatbot analyzes user messages, identifies the user's intent, and provides an appropriate response.

---

## 📌 Project Overview

This project is a simple Artificial Intelligence chatbot that uses **Natural Language Processing (NLP)** and a **Deep Learning model** to understand user queries.

The chatbot can recognize different variations of similar questions and classify them into the correct intent.

For example:

- "What's up?"
- "What are you doing?"
- "wt r u doing?"

These different inputs can be understood as similar intents and the chatbot provides an appropriate response.

---

## 🎯 Problem Statement

Users often need quick responses to common questions. Traditional systems may require navigating through menus or waiting for human assistance.

This project solves this problem by creating an automated chatbot that:

- Accepts user input
- Processes the text using NLP techniques
- Identifies the user's intent
- Generates an appropriate predefined response

The chatbot provides instant automated responses through both a **command-line interface** and a **Streamlit web application**.

---

## ✨ Features

- 🤖 Automated chatbot responses
- 🧠 Intent classification using Deep Learning
- 💬 Understands different variations of user queries
- 🔤 Text preprocessing using NLP
- ✂️ Tokenization
- 📖 Lemmatization
- 📊 Bag-of-Words representation
- 🧠 Neural Network-based intent prediction
- 💻 Command-line chatbot interface
- 🌐 Interactive web interface using Streamlit
- 🚀 Deployment-ready application

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Building and training the neural network |
| NLTK | Natural Language Processing |
| NumPy | Numerical operations |
| Pickle | Saving processed words and classes |
| Streamlit | Web application interface |
| Git & GitHub | Version control and project hosting |

---

## 🧠 How the Chatbot Works

User Input
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
Lemmatization
    ↓
Bag of Words
    ↓
Trained Neural Network
    ↓
Intent Prediction
    ↓
Appropriate Response


## 📂 Project Structure

nlp-chatbot/
│
├── data/
│   └── intents.json
│
├── app.py
├── chat.py
├── main.py
├── chatbot_model.h5
├── classes.pkl
├── words.pkl
├── requirements.txt
├── .gitignore
└── README.md


## ⚙️ Installation

### 1. Clone the Repository


git clone https://github.com/rajitha481/nlp-chatbot.git


### 2. Navigate to the Project Folder


cd nlp-chatbot

### 3. Create a Virtual Environment

python -m venv .venv


### 4. Activate the Virtual Environment

#### Windows


.venv\Scripts\activate

### 5. Install Dependencies


pip install -r requirements.txt



## 🏋️ Train the Model

Run the following command:


python main.py


The training process will generate the following files:


chatbot_model.h5
words.pkl
classes.pkl



## 💬 Run the Chatbot

To run the command-line chatbot:


python chat.py


Example:


Chatbot is ready!
Type 'quit' to exit.

You: hello
Bot: Hey! Good to see you.

You: what's up
Bot: I'm here chatting with you!

You: quit


## 🌐 Run the Streamlit Web Application

Run:


streamlit run app.py


The application will open in your browser.

Usually at:


http://localhost:8501


## 🔤 NLP Techniques Used

### 1. Tokenization

Tokenization breaks a sentence into individual words.

Example:


"What are you doing?"

↓

["What", "are", "you", "doing"]

### 2. Lemmatization

Lemmatization converts words into their base form.

Example:


running → run
playing → play


### 3. Bag of Words

The processed text is converted into a numerical representation that can be understood by the machine learning model.

### 4. Intent Classification

A TensorFlow/Keras neural network predicts the user's intent based on the processed input and selects an appropriate response.



## 🚀 Future Improvements

- Add more intents and training data
- Improve chatbot accuracy
- Add confidence scores for predictions
- Store chat history
- Add user authentication
- Connect the chatbot to a database
- Integrate Generative AI or Large Language Models
- Add voice input and speech responses
- Improve the Streamlit user interface

---

## 📸 Example Interaction

You: Hello
Bot: Hey! Good to see you.

You: What's up?
Bot: I'm here chatting with you!

You: What are you doing?
Bot: I'm helping you with your questions.


## 👩‍💻 Author

**Rajitha**

GitHub: [https://github.com/rajitha481](https://github.com/rajitha481)

---

## ⭐ Support

If you like this project, consider giving the repository a **star ⭐**!
