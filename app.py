"""
IMDB Sentiment Analysis API
Serves the trained Keras model as a web API.
"""
import re
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI
from pydantic import BaseModel

# Create the app
app = FastAPI(title="IMDB Sentiment API")

# Load the trained model once, when the app starts
model = keras.models.load_model("imdb_sentiment_model.keras")


# Define the shape of the incoming request (a JSON body with a "text" field)
class Review(BaseModel):
    text: str


def clean(text: str) -> str:
    """Remove <br /> tags, matching how the model was trained."""
    return re.sub(r"<br\s*/?>", " ", text)


@app.get("/")
def home():
    """A simple health-check so you can see the API is alive in a browser."""
    return {"message": "IMDB Sentiment API is running. Send a POST to /predict"}


@app.post("/predict")
def predict(review: Review):
    """Take a review, return the predicted sentiment and confidence."""
    cleaned = clean(review.text)
    pred = model.predict(tf.constant([cleaned]))
    score = float(pred[0][0])
    return {
        "text": review.text,
        "sentiment": "positive" if score > 0.5 else "negative",
        "confidence": round(score, 3),
    }
