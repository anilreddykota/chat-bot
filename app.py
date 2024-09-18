import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
CORS(app)

# Global Variables to Cache Model and Embeddings
tokenizer = None
model = None
question_embeddings = None
questions = []
answers = []

# Function to load data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'amiledata.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    df = pd.read_json(file_path)
    return df['question'].tolist(), df['answer'].tolist()

# Function to load BERT tokenizer and model
def load_bert_model():
    global tokenizer, model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

# Function to get embeddings using BERT
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Precompute and Cache Question Embeddings
def cache_question_embeddings():
    global question_embeddings
    question_embeddings = get_embeddings(questions)

# Function to find the closest question using cosine similarity
def find_closest_question(user_question, threshold=0.5):
    user_embedding = get_embeddings([user_question])
    similarities = cosine_similarity(user_embedding, question_embeddings)

    max_similarity_index = similarities.argmax()
    max_similarity_score = similarities.max()

    if max_similarity_score >= threshold:
        return answers[max_similarity_index]
    else:
        return None

# Main function to generate a response based on user input
def generate_response(user_question):
    answer = find_closest_question(user_question)
    
    if answer is None:
        return "I'm sorry, but I don't have a specific answer for that question. Can you please rephrase or ask something else?"
    else:
        return answer

@app.route('/chat', methods=['POST'])  # POST is more appropriate for chat
def chat():
    data = request.json
    user_question = data.get('question', '').strip()

    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    # Generate answer based on user question
    answer = generate_response(user_question)
    return jsonify({'answer': answer})

# Initialize the app with preloaded data and embeddings
@app.before_first_request
def initialize():
    global questions, answers
    # Load model and tokenizer once
    load_bert_model()
    
    # Load the questions and answers from JSON file
    try:
        questions, answers = load_data()
    except FileNotFoundError as e:
        questions, answers = [], []
        print(str(e))
    
    # Cache the question embeddings once at startup
    cache_question_embeddings()

# Vercel requires the app to be callable as 'app'
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
