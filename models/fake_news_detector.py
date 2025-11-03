import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import config
import streamlit as st
import re

class FakeNewsDetector:
    def __init__(self):
        self.model_path = config.MODELS_DIR / "fake_news_detector.pkl"
        self.vectorizer_path = config.MODELS_DIR / "fake_news_vectorizer.pkl"
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
            st.error(f"Error loading fake news model: {e}")
            self.train_model()
    
    def extract_features(self, text):
        """Extract additional features from text"""
        features = {}
        
        # Text length
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        # Punctuation and capitalization
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        # Sensational words
        sensational_words = ['shocking', 'unbelievable', 'amazing', 'incredible', 'you won\'t believe']
        features['sensational_count'] = sum(word.lower() in text.lower() for word in sensational_words)
        
        return features
    
    def train_model(self):
        """Train fake news detection model with sample data"""
        # Sample training data
        real_news = [
            "The government announced new economic policies aimed at reducing inflation and supporting small businesses.",
            "Scientists have published a peer-reviewed study showing promising results in cancer treatment research.",
            "The stock market experienced moderate gains today following positive employment data.",
            "Local authorities report successful completion of infrastructure development project.",
            "Research team from university discovers new method for water purification."
        ]
        
        fake_news = [
            "SHOCKING! Celebrity reveals government conspiracy that will change EVERYTHING!",
            "You won't believe what scientists are hiding from you! Click now!",
            "BREAKING: Miracle cure discovered! Doctors hate this ONE simple trick!",
            "UNBELIEVABLE truth about vaccines that THEY don't want you to know!!!",
            "AMAZING secret the government doesn't want revealed! Share before deleted!"
        ]
        
        # Prepare data
        texts = real_news + fake_news
        labels = [0] * len(real_news) + [1] * len(fake_news)  # 0 = real, 1 = fake
        
        # Create vectorizer
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, labels)
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def predict(self, text):
        """Predict if news is fake"""
        if not self.model or not self.vectorizer:
            raise ValueError("Model not loaded or trained")
        
        # Transform text
        X = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        fake_probability = float(probabilities[1])  # Probability of being fake
        
        # Extract text features for analysis
        text_features = self.extract_features(text)
        
        # Determine verdict
        if fake_probability > 0.7:
            verdict = "Likely Fake"
        elif fake_probability > 0.4:
            verdict = "Suspicious"
        else:
            verdict = "Likely Real"
        
        return {
            'is_fake': bool(prediction),
            'fake_probability': fake_probability,
            'real_probability': float(probabilities[0]),
            'verdict': verdict,
            'text_features': text_features
        }
    
    def analyze_credibility(self, text, url=None):
        """Comprehensive credibility analysis"""
        result = self.predict(text)
        
        # Additional checks
        credibility_score = 100 - (result['fake_probability'] * 100)
        
        warnings = []
        if result['text_features']['sensational_count'] > 2:
            warnings.append("High use of sensational language")
        if result['text_features']['exclamation_count'] > 3:
            warnings.append("Excessive exclamation marks")
        if result['text_features']['caps_ratio'] > 0.3:
            warnings.append("Excessive capitalization")
        
        result['credibility_score'] = credibility_score
        result['warnings'] = warnings
        
        return result