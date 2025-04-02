import streamlit as st
import torch
import numpy as np
import joblib
import catboost
import lightgbm as lgb
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel

# Load trained models
log_reg = joblib.load("./models/logistic_regression.pkl")
dec_tree = joblib.load("./models/decision_tree.pkl")
rand_forest = joblib.load("./models/random_forest.pkl")
xgb_model = joblib.load("./models/xgboost_model.pkl")
cat_model = catboost.CatBoostClassifier()
cat_model.load_model("./models/catboost_model.cbm")
lgb_model = lgb.Booster(model_file="./models/lightgbm_model.txt")

# Load DeBERTa tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
bert_model = AutoModel.from_pretrained("microsoft/deberta-v3-base")

# Predefined strong negative phrases
negative_words = {"frustrated", "angry", "disappointed", "upset", "hate", "bad", "sad", "terrible", "awful"}
negation_words = {"not", "no", "never", "n't"}

# Function to preprocess text
def preprocess_text(text):
    words = text.split()
    for i in range(len(words) - 1):
        if words[i] in negation_words:
            words[i] = words[i] + "_" + words[i+1]  # Merge negation words
            words[i+1] = ""  # Remove duplicate
    return " ".join([w for w in words if w]).strip()

# Function to get DeBERTa embeddings
def get_embedding(text):
    text = preprocess_text(text)
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze().numpy()

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    embedding = get_embedding(processed_text).reshape(1, -1)

    # Get predictions
    log_pred = log_reg.predict(embedding)[0]
    dt_pred = dec_tree.predict(embedding)[0]
    rf_pred = rand_forest.predict(embedding)[0]
    xgb_pred_prob = xgb_model.predict(embedding)[0]
    cat_pred = cat_model.predict(embedding)[0]
    lgb_pred_prob = lgb_model.predict(embedding)[0]

    # Convert probability outputs to binary
    xgb_pred = 1 if xgb_pred_prob > 0.55 else 0  # Adjust threshold
    lgb_pred = 1 if lgb_pred_prob > 0.55 else 0  # Adjust threshold

    # Manual override for strong negative words
    for word in processed_text.split():
        if word in negative_words:
            log_pred = dt_pred = rf_pred = xgb_pred = cat_pred = lgb_pred = 0

    return {
        "Logistic Regression": "Positive" if log_pred == 1 else "Negative",
        "Decision Tree": "Positive" if dt_pred == 1 else "Negative",
        "Random Forest": "Positive" if rf_pred == 1 else "Negative",
        "XGBoost": "Positive" if xgb_pred == 1 else "Negative",
        "CatBoost": "Positive" if cat_pred == 1 else "Negative",
        "LightGBM": "Positive" if lgb_pred == 1 else "Negative"
    }

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a sentence to analyze sentiment using various ML and Modern Ensemble models.")

user_input = st.text_input("Enter a sentence:", "I am very happy today")

if st.button("Analyze Sentiment"):
    results = predict_sentiment(user_input)
    
    st.subheader("Prediction Results:")
    for model, sentiment in results.items():
        st.write(f"**{model}**: {sentiment}")

#  python -m streamlit run app.py         