# IMDB Movie Review Sentiment Analysis — End-to-End ML Project

An end-to-end natural language processing project that classifies movie reviews as **positive** or **negative**, built with Python, TensorFlow/Keras, and Microsoft Azure. The trained model is served as a live REST API deployed on Azure App Service.

## Live demo

The trained model served as a live API on Azure, predicting sentiment in real time:

![Live API prediction](screenshots/prediction.png)

Model accuracy computed directly in SQL against the stored predictions:

![Model accuracy](screenshots/accuracy.png)

## What it does

1. Takes raw IMDB movie reviews and cleans the text
2. Trains a neural network to predict sentiment (positive / negative)
3. Stores the reviews and model predictions in a cloud **Azure SQL Database**
4. Serves the model as a live **REST API** (FastAPI) deployed on Azure

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python |
| Modelling | TensorFlow / Keras |
| Data handling | pandas |
| Cloud database | Azure SQL Database |
| DB connection | pyodbc |
| API | FastAPI + Uvicorn |
| Deployment | Azure App Service |

## Project structure

- `IMDB_sentiment_model.ipynb` — loads data, cleans text, builds & trains the neural network, saves the model
- `azure_database.ipynb` — connects to Azure SQL, uploads reviews, runs predictions, writes results back, computes accuracy in SQL
- `app.py` — FastAPI application that serves the model as a `/predict` endpoint
- `requirements.txt` — Python dependencies
- `imdb_sentiment_model.keras` — the trained model
- `.env.example` — template for the database credentials (copy to `.env` and fill in)

## The data

This project uses the **IMDB Dataset of 50K Movie Reviews** (25,000 positive, 25,000 negative).
Download it from Kaggle: search "IMDB Dataset of 50K Movie Reviews" by Lakshmipathi N.
Place `IMDB Dataset.csv` in the project folder. (The dataset is not included in this repo because of its size.)

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Azure SQL credentials.
3. Run the notebooks in order: `IMDB_sentiment_model.ipynb`, then `azure_database.ipynb`.
4. To serve the model as an API locally:
   ```
   uvicorn app:app --reload
   ```
   Then open `http://127.0.0.1:8000/docs` to test predictions interactively.

## Results

- The model reaches roughly **85% accuracy** on held-out reviews.
- Accuracy is verified directly in SQL by comparing the true labels against the model's predictions stored in the database.

## Key learnings

- Building a full pipeline from raw data to a deployed cloud service
- Model serialization and the importance of consistent text cleaning between training and inference
- Working with a serverless cloud database (connection handling, auto-pause behaviour)
- Deploying a memory-heavy ML model and choosing an appropriate hosting tier
- Keeping credentials out of source control using environment variables

## Notes

- The database uses Azure's free serverless tier with auto-pause to stay within free limits.
- Credentials are stored in a local `.env` file (excluded from version control via `.gitignore`).
