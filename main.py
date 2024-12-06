import os
import json
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
from dotenv import load_dotenv
import openai
import torch
from transformers import BertTokenizer
import re
import emoji
from bs4 import BeautifulSoup
import torch.nn as nn
from transformers import BertModel

# Konfigurasi
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

load_dotenv()  # Muat variabel dari .env
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ambil API key

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ubah jika perlu untuk membatasi origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pembersihan teks
def clean_text(text):
    text = str(text)
    text = emoji.demojize(text)
    text = re.sub(r'\:(.*?)\:', '', text)
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text

# Model klasifikasi berbasis BERT
class BertClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5, num_classes=6):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.fc1(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        return self.fc2(pooled_output)

# Inisialisasi model dan tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertClassifier(num_classes=6)
model.load_state_dict(torch.load('bert_model.pth', map_location=torch.device('cpu'), weights_only=True), strict=False)
model.eval()

emotion_labels = ['caring', 'love', 'gratitude', 'sadness', 'fear', 'anger']

# Fungsi prediksi
def predict_emotions(text, threshold=0.6):
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probabilities = torch.sigmoid(outputs)
        predicted_classes = (probabilities > threshold).int()
        predicted_emotions = [emotion_labels[i] for i, val in enumerate(predicted_classes[0]) if val == 1]
        return {
            "predicted_emotions": predicted_emotions,
            "probabilities": probabilities.tolist()
        }

# Schema untuk request body
class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PredictRequest):
    text = request.text

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        prediction = predict_emotions(text)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process prediction: {str(e)}")

@app.get("/suggestions")
async def get_suggestions(emotion: str = Query(...), text: str = Query(...)):
    if not emotion or not text:
        raise HTTPException(status_code=400, detail="Emotion and text are required")

    prompt = f"Based on the predicted emotion: {emotion}, and the issue: {text}, provide suggestions to improve mental well-being."

    try:
        # Memanggil OpenAI Chat API
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.85
        )
        suggestion = gpt_response['choices'][0]['message']['content'].strip()
        suggestions_array = suggestion.split('\n')

        return {"suggestions": suggestions_array}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")
        