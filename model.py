from transformers import DistilBertModel, DistilBertTokenizer
import torch

# Load pretrained tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Save the tokenizer and model
tokenizer.save_pretrained('./local_model/tokenizer')
model.save_pretrained('./local_model/model')
