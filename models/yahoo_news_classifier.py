import pickle
import sys
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC # Using LinearSVC as per the training script
import config
import streamlit as st # Assuming Streamlit might be used for UI/error reporting

# Define the category map based on the training script
YAHOO_CATEGORIES_MAP = {
    1: 'Society & Culture', 2: 'Science & Mathematics', 3: 'Health',
    4: 'Education & Reference', 5: 'Computers & Internet', 6: 'Sports',
    7: 'Business & Finance', 8: 'Entertainment & Music', 9: 'Family & Relationships',
    10: 'Politics & Government'
}
YAHOO_CATEGORIES = list(YAHOO_CATEGORIES_MAP.values())

class YahooAnswersClassifier:
    """
    Clasificator pentru setul de date Yahoo! Answers Topics.
    
    Încărcă un model LinearSVC și un vectorizer TF-IDF antrenat 
    (presupunând că au fost salvați de scriptul de antrenare).
    """
    def __init__(self):
        # Paths based on the saving step in the training script
        self.model_path = config.MODELS_DIR / "yahoo_answers_classifier_linsvc.pkl"
        self.vectorizer_path = config.MODELS_DIR / "yahoo_answers_vectorizer.pkl"
        self.categories = YAHOO_CATEGORIES
        self.model = None
        self.vectorizer = None
        # LinearSVC does not directly provide predict_proba, 
        # so we need to enable it if the model was not trained with 'probability=True' 
        # (which is slow/complex). We will adjust the predict method.
        
        self.load_model() # We assume the model is pre-trained by the other script

    def load_model(self):
        """
        Încarcă modelul și vectorizer-ul antrenat.
        
        NOTĂ: Nu include logica de antrenare ('train_model') în clasa de inferență,
        deoarece setul de date Yahoo! Answers este foarte mare și antrenarea ar trebui 
        să fie o etapă separată, pre-calculată.
        """
        if self.model_path.exists() and self.vectorizer_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                # Check if LinearSVC was trained with probability support
                if not hasattr(self.model, 'predict_proba'):
                    # LinearSVC doesn't support predict_proba by default (unless probability=True is set 
                    # during training, which creates a wrapper). We'll handle this in predict().
                    print("AVERTISMENT: LinearSVC nu are predict_proba. Nu se vor returna confidențe.")
                    self.supports_proba = False
                else:
                     self.supports_proba = True
                
                print("Model Yahoo Answers și Vectorizer încărcate cu succes.")
            except Exception as e:
                # Use st.error if running in Streamlit context
                if 'st' in sys.modules: 
                    st.error(f"Eroare la încărcarea modelului/vectorizer-ului Yahoo: {e}")
                else:
                    print(f"Eroare la încărcarea modelului/vectorizer-ului Yahoo: {e}")
                raise RuntimeError("Nu s-a putut încărca modelul. Asigură-te că antrenarea a rulat cu succes.")
        else:
            raise FileNotFoundError(
                "Fișierele modelului Yahoo! Answers nu au fost găsite. "
                "Rulează scriptul de antrenare ('yahoo_logistic_classifier.py') mai întâi."
            )

    def predict(self, text):
        """
        Prezice categoria pentru un text dat (combinația de titlu, conținut, răspuns).
        Returnează predicția, confidența (0.0 dacă nu e disponibilă) și scorurile categoriilor.
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Modelul nu este încărcat sau antrenat.")
        
        # 1. Transformă textul
        X = self.vectorizer.transform([text])
        
        # 2. Prezice
        prediction = self.model.predict(X)[0]
        
        confidence = 0.0
        category_scores = {cat: 0.0 for cat in self.categories}
        
        # 3. Calculează confidența/scorurile (LinearSVC nu are predict_proba nativ)
        # Dacă modelul a fost antrenat cu probabilități, folosește-le.
        # Altfel, putem folosi decision_function.
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(max(probabilities))
            category_scores = {
                category: float(prob) 
                for category, prob in zip(self.model.classes_, probabilities)
            }
        elif hasattr(self.model, 'decision_function'):
             # Decision function gives a measure of distance from the hyperplane.
             # We can't easily convert this to a probability/confidence score.
             # For simplicity, we just set confidence to 0.0 or could use a normalization
             # method (e.g., softmax) which is out of scope here.
             pass
        
        return prediction, confidence, category_scores
    
    def batch_predict(self, texts):
        """Prezice categorii pentru texte multiple."""
        results = []
        for text in texts:
            try:
                # Concatenează textul într-o singură intrare, 
                # deoarece modelul este antrenat pe titlu + conținut + răspuns
                # *NOTĂ: În mod ideal, input-ul ar trebui să fie de forma (titlu, conținut, răspuns)
                # dar pentru simplitate, folosește textul ca input consolidat.*
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

# # Exemplu de utilizare (necesită config și fișierele modelului)
# if __name__ == "__main__":
# # Presupunând că 'config' este disponibil
# try:
#     classifier = YahooAnswersClassifier()
    
#     # Textul ar trebui să fie o combinație de titlu, conținut și cel mai bun răspuns 
#     # (așa cum este antrenat modelul)
#     test_text = (
#         "Why is the sky blue? Is it because of the water in the atmosphere? "
#         "The sky is blue because of Rayleigh scattering. Sunlight hits the "
#         "Earth's atmosphere and blue light is scattered more than other colors."
#     )
    
#     prediction, confidence, scores = classifier.predict(test_text)
    
#     print("\n--- TEST PREDICTIE ---")
#     print(f"Text: '{test_text[:70]}...'")
#     print(f"Predicție: **{prediction}**")
#     print(f"Confidență: {confidence:.4f} (Notă: 0.0 pentru LinearSVC fără predict_proba)")
#     print(f"Scoruri: {scores}")
    
# except FileNotFoundError as e:
#     print(f"\nEROARE: {e}")
# except Exception as e:
#     print(f"\nEROARE NECESARĂ: {e}")