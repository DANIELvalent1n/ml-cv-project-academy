import pickle
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import config  # Asigură-te că acest fișier de configurare este disponibil

# Adaugă import pentru Streamlit doar dacă intenționezi să folosești clasa într-o aplicație Streamlit
# import streamlit as st

class TextOriginClassifier:
    """
    Clasificator pentru a determina dacă un eseu este scris de om (0) sau generat de mașină (1).
    Folosește un model LogisticRegression și vectorizare TF-IDF.
    """
    
    def __init__(self, data_path="data/datasets/Training_Essay_Data.csv"):
        """
        Inițializează căile și încarcă sau antrenează modelul.
        """
        self.model_path = config.MODELS_DIR / "origin_classifier_model.pkl"
        self.vectorizer_path = config.MODELS_DIR / "origin_vectorizer.pkl"
        self.data_path = data_path
        self.model = None
        self.vectorizer = None
        self.classes = ['Human Written (0)', 'AI Generated (1)'] # Etichetele corespunzătoare pentru 0 și 1
        self.load_or_train()
    
    # --- Metode de Încărcare/Antrenare ---
    
    def load_or_train(self):
        """Încarcă modelul și vectorizatorul existente sau antrenează altele noi."""
        if self.model_path.exists() and self.vectorizer_path.exists():
            print("Încărc modelul și vectorizatorul existente...")
            self.load_model()
        else:
            print("Fișierele modelului nu au fost găsite. Încep antrenarea...")
            self.train_model()
    
    def load_model(self):
        """Încarcă modelul antrenat și vectorizatorul de pe disc."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            # Dacă folosești Streamlit, poți înlocui print cu: st.success("Model și vectorizator încărcate cu succes.")
        except Exception as e:
            # Dacă încarcarea eșuează, încearcă să antrenezi din nou
            print(f"Eroare la încărcarea modelului: {e}. Încep antrenarea din nou...")
            # Dacă folosești Streamlit, poți folosi: st.error(f"Eroare la încărcarea modelului: {e}")
            self.train_model()

    def train_model(self):
        """Antrenează modelul de clasificare folosind datele din Training_Essay_Data.csv."""
        # 1. Încărcarea Setului de Date
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Eroare: Fișierul de date '{self.data_path}' nu a fost găsit.")

        # 2. Definirea Caracteristicilor (X) și a Țintei (y)
        X = df['text']
        y = df['generated']
        
        # 3. Împărțirea Datelor (80/20) - necesar pentru 'fit' pe TF-IDF și LogisticRegression
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 4. Extragerea de Caracteristici (Vectorizare TF-IDF)
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # 5. Antrenarea Modelului
        self.model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
        self.model.fit(X_train_tfidf, y_train)
        
        # 6. Salvarea Modelului și a Vectorizatorului
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True) # Asigură-te că directorul există
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
        print("Model și vectorizator antrenați și salvați cu succes!")
        
    # --- Metoda de Predicție ---

    def predict(self, text: str):
        """
        Prezice originea (om sau mașină) și încrederea pentru un singur text.
        
        Returnează:
            prediction (str): Eticheta prezisă (ex: 'Generat de Mașină (1)')
            confidence (float): Încrederea în predicție (probabilitatea maximă)
            category_scores (dict): Scorurile de încredere pentru fiecare clasă
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Modelul nu este încărcat sau antrenat.")
        
        # Transformă textul folosind vectorizatorul antrenat
        X = self.vectorizer.transform([text])
        
        # Prezice clasa și probabilitățile
        prediction_label = self.model.predict(X)[0] # 0 sau 1
        probabilities = self.model.predict_proba(X)[0] # [prob_clasa0, prob_clasa1]
        
        # Maparea predicției numerice la eticheta de text
        prediction_text = self.classes[prediction_label]
        
        # Obține scorurile de încredere pentru toate categoriile
        category_scores = {
            self.classes[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        confidence = float(max(probabilities))
        
        return prediction_text, confidence, category_scores
    
    def batch_predict(self, texts: list):
        """
        Prezice categoriile pentru o listă de texte.
        """
        results = []
        for text in texts:
            try:
                pred, conf, scores = self.predict(text)
                results.append({
                    'text': text,
                    'origin': pred,
                    'confidence': conf,
                    'scores': scores
                })
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        return results

# Exemplu de utilizare (opțional, în același fișier sau separat)
# if __name__ == "__main__":
#     # Trebuie să te asiguri că ai un 'config.py' cu 'MODELS_DIR' definit
#     # și fișierul de date la "data/datasets/Training_Essay_Data.csv"
#     
#     # Exemplu de configurare simplă pentru test (ar trebui să fie într-un fișier config.py real):
#     # class Config:
#     #     MODELS_DIR = Path("models")
#     # config = Config()

#     classifier = TextOriginClassifier()
#     
#     sample_text_human = "Am scris acest eseu concentrându-mă pe detalii și pe un ton personal. Structura este complexă și fluidă, reflectând gândirea mea critică."
#     sample_text_machine = "Un model lingvistic mare a generat acest text. Acesta urmează o structură strictă, utilizează un vocabular formal și este lipsit de erori neintenționate."
#     
#     pred_h, conf_h, scores_h = classifier.predict(sample_text_human)
#     print(f"\nText: '{sample_text_human[:50]}...'")
#     print(f"Predicție: {pred_h}, Încredere: {conf_h:.4f}")
#     print(f"Scoruri: {scores_h}")

#     pred_m, conf_m, scores_m = classifier.predict(sample_text_machine)
#     print(f"\nText: '{sample_text_machine[:50]}...'")
#     print(f"Predicție: {pred_m}, Încredere: {conf_m:.4f}")
#     print(f"Scoruri: {scores_m}")