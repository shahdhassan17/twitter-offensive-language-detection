from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import joblib
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="Twitter Sentiment Analysis API"
)

templates = Jinja2Templates(directory="templates")

# =========================
# Load Model
# =========================

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# =========================
# NLP Tools
# =========================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

LABELS = {
    0: "Normal",
    1: "Offensive"
}

# =========================
# Text Cleaning
# =========================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)

# =========================
# API Schema
# =========================

class TweetRequest(BaseModel):
    text: str

# =========================
# Web UI
# =========================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "text": ""
        }
    )

@app.post("/", response_class=HTMLResponse)
def predict_web(
    request: Request,
    text: str = Form(...)
):

    cleaned = clean_text(text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    label = LABELS[int(prediction)]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": label,
            "text": text
        }
    )

# =========================
# REST API Endpoint
# =========================

@app.post("/predict")
def predict(data: TweetRequest):

    cleaned = clean_text(data.text)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    return {
        "text": data.text,
        "prediction": LABELS[int(prediction)]
    }
