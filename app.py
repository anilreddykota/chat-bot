import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
CORS(app)

# Function to load data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'amiledata.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    df = pd.read_json(file_path)
    return df['question'].tolist(), df['answer'].tolist()

# Load the questions and answers
try:
    questions, answers = load_data()
except FileNotFoundError as e:
    questions, answers = [], []
    print(str(e))

# Function to load BERT tokenizer and model
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

# Function to get embeddings using BERT
def get_embeddings(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Cache question embeddings
def get_question_embeddings(tokenizer, model):
    return get_embeddings(questions, tokenizer, model)

# Function to find the closest question using cosine similarity
def find_closest_question(user_question, tokenizer, model, question_embeddings, threshold=0.5):
    user_embedding = get_embeddings([user_question], tokenizer, model)
    similarities = cosine_similarity(user_embedding, question_embeddings)

    max_similarity_index = similarities.argmax()
    max_similarity_score = similarities.max()

    if max_similarity_score >= threshold:
        return answers[max_similarity_index]
    else:
        return None

# Main function to generate a response based on user input
def generate_response(user_question, tokenizer, model, question_embeddings):
    answer = find_closest_question(user_question, tokenizer, model, question_embeddings)
    
    if answer is None:
        return "I'm sorry, but I don't have a specific answer for that question. Can you please rephrase or ask something else?"
    else:
        return answer

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get('question', '')

    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    # Load model and tokenizer for every request (can optimize for production)
    tokenizer, model = load_bert_model()
    
    # Get question embeddings
    question_embeddings = get_question_embeddings(tokenizer, model)
    
    # Generate answer based on user question
    answer = generate_response(user_question, tokenizer, model, question_embeddings)
    return jsonify({'answer': answer})

# Vercel requires the app to be callable as 'app'
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
