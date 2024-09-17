from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

app = Flask(_name_)  # Fix for Flask app initialization
CORS(app)

def load_data():
    df = pd.read_json('amiledata.json')  # Make sure the path is correct when deploying
    return df['question'].tolist(), df['answer'].tolist()

questions, answers = load_data()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

question_embeddings = get_embeddings(questions)

def find_closest_question(user_question, threshold=0.5):
    user_embedding = get_embeddings([user_question])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    
    max_similarity_index = similarities.argmax()
    max_similarity_score = similarities.max()

    if max_similarity_score >= threshold:
        return answers[max_similarity_index]
    else:
        return None

def generate_response(user_question):
    answer = find_closest_question(user_question)
    
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
    
    answer = generate_response(user_question)
    return jsonify({'answer': answer})

# Vercel requires the app to be callable as 'app'
def handler(event, context):
    return app(event, context)

if _name_ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
