# yahoo_logistic_classifier.py

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import config
import csv

import config

def train_yahoo_answers_classifier():
    """
    Antrenează clasificatorul pe Yahoo! Answers Topics Dataset folosind LinearSVC.
    """
    print("=" * 60)
    print("ANTRENARE YAHOO! ANSWERS CLASSIFIER CU LogisticRegression") 
    print("=" * 60)
    
    try:
        # 1. Încărcare dataset
        print("\n1. Încărcare dataset Yahoo! Answers Topics...")
        from datasets import load_dataset
        
        # Descarcă și încarcă dataset
        # Setul de date Yahoo! Answers este foarte mare (peste 1.4 milioane de mostre)
        #dataset = load_dataset("yahoo_answers_topics")

        train_path = config.DATASETS_DIR / "train.csv"
        test_path = config.DATASETS_DIR / "test.csv"
        
        # Extrage datele
        train_data = load_dataset("csv", data_files=str(train_path), split='train')
        test_data = load_dataset("csv", data_files=str(test_path), split='train')
        
        print(f"  ✓ Training samples: {len(train_data)}")
        print(f"  ✓ Test samples: {len(test_data)}")
        
        # 2. Pregătire date
        print("\n2. Pregătire date...")
        
        # Setul de date conține coloanele 'question_title', 'question_content' și 'best_answer'.
        # Vom concatena toate câmpurile text pentru a obține o reprezentare mai bogată.
        X_train = [
            f"{item['question_title']} {item['question_content']} {item['best_answer']}" 
            for item in train_data
        ]
        y_train = [item['topic'] for item in train_data]
        
        X_test = [
            f"{item['question_title']} {item['question_content']} {item['best_answer']}" 
            for item in test_data
        ]
        y_test = [item['topic'] for item in test_data]
        
        # Mapare labels (1-10) la categorii (specific Yahoo! Answers)
        label_map = {
            1: 'Society & Culture', 2: 'Science & Mathematics', 3: 'Health',
            4: 'Education & Reference', 5: 'Computers & Internet', 6: 'Sports',
            7: 'Business & Finance', 8: 'Entertainment & Music', 9: 'Family & Relationships',
            10: 'Politics & Government'
        }
        y_train_mapped = [label_map[label] for label in y_train]
        y_test_mapped = [label_map[label] for label in y_test]
        
        print(f"  ✓ Număr de categorii: {len(label_map)}")
        print(f"  ✓ Categorii: {list(label_map.values())}")
        
        # 3. Creare vectorizer
        print("\n3. Creare TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=75000,          # Mărit la 100k
            stop_words='english',
            ngram_range=(1, 2),           # Trecut la (1, 3)
            min_df=2,
            max_df=0.8,
            sublinear_tf=True             # NOU: Aplică scalare logaritmică
            )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        print(f"  ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # 4. Antrenare model
        print("\n4. Antrenare model Linear Support Vector Classifier (LogReg)...")
        # LinearSVC este o alegere excelentă pentru clasificarea textului la scară largă
        model = LinearSVC(
            dual="auto", 
            C=0.8,                # NOU: Regularizare ușor crescută
            loss='hinge',         # NOU: Schimbă funcția de pierdere
            random_state=42, 
            verbose=1,
            max_iter=5000         # Poate ajuta la convergență pe seturi mari
            )
        
        model.fit(X_train_vec, y_train_mapped)
        
        # 5. Evaluare
        print("\n5. Evaluare model...")
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test_mapped, y_pred)
        
        print(f"\n  ✓ Accuracy: {accuracy*100:.2f}%")
        print("\n  Classification Report:")
        print(classification_report(y_test_mapped, y_pred))
        
        # 6. Salvare model
        print("\n6. Salvare model...")
        model_path = config.MODELS_DIR / "yahoo_answers_classifier_linsvc.pkl" 
        vectorizer_path = config.MODELS_DIR / "yahoo_answers_vectorizer.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print(f"  ✓ Model salvat: {model_path}")
        print(f"  ✓ Vectorizer salvat: {vectorizer_path}")
        
        print("\n✅ ANTRENARE CLASSIFIER COMPLETĂ!")
        return True
        
    except Exception as e:
        print(f"\n❌ EROARE: {e}")
        print("\nAsigură-te că ai instalat pachetele necesare: pip install datasets scikit-learn")
        return False

def main():
    train_yahoo_answers_classifier()

if __name__ == "__main__":
    main()