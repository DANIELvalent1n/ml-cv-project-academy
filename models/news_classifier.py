import pickle
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import config
import streamlit as st

class NewsClassifier:
    def __init__(self):
        self.model_path = config.MODELS_DIR / "news_classifier.pkl"
        self.vectorizer_path = config.MODELS_DIR / "news_vectorizer.pkl"
        self.categories = config.NEWS_CATEGORIES
        self.model = None
        self.vectorizer = None
        self.load_or_train()
    
    def load_or_train(self):
        """Load existing model or train new one"""
        if self.model_path.exists() and self.vectorizer_path.exists():
            self.load_model()
        else:
            self.train_model()
    
    def load_model(self):
        """Load trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.train_model()

    def train_model(self):
        """Train classification model using AG News dataset"""
        # Load AG News dataset
        dataset = load_dataset("ag_news")
        
        # Convert labels to categories (AG News uses indices 0-3)
        label_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        
        # Prepare training data
        train_texts = dataset["train"]["text"]
        train_labels = [label_map[label] for label in dataset["train"]["label"]]
        
        # Create and train vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = self.vectorizer.fit_transform(train_texts)
        
        # Train model
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X, train_labels)
        
        # Save model and vectorizer
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def predict(self, text):
        """Predict category for text"""
        if not self.model or not self.vectorizer:
            raise ValueError("Model not loaded or trained")
        
        # Transform text
        X = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get confidence scores for all categories
        category_scores = {
            category: float(prob) 
            for category, prob in zip(self.model.classes_, probabilities)
        }
        
        confidence = float(max(probabilities))
        
        return prediction, confidence, category_scores
    
    def batch_predict(self, texts):
        """Predict categories for multiple texts"""
        results = []
        for text in texts:
            try:
                pred, conf, scores = self.predict(text)
                results.append({
                    'text': text,
                    'category': pred,
                    'confidence': conf,
                    'scores': scores
                })
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        return results