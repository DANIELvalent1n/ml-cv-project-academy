import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Încărcarea Setului de Date
# Asigură-te că fișierul 'Training_Essay_Data.csv' se află în directorul de lucru.
try:
    df = pd.read_csv("data/datasets/Training_Essay_Data.csv")
except FileNotFoundError:
    print("Eroare: Fișierul 'Training_Essay_Data.csv' nu a fost găsit.")
    exit()

# 2. Definirea Caracteristicilor (X) și a Țintei (y)
# 'text' este coloana de intrare (eseul), iar 'generated' este ținta (0=uman, 1=mașină).
X = df['text']
y = df['generated']

# 3. Împărțirea Datelor în Seturi de Antrenare și Testare (80/20)
# 'stratify=y' asigură că proporția claselor (0 și 1) este menținută în ambele seturi.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Număr de eșantioane de antrenare: {len(X_train)}")
print(f"Număr de eșantioane de testare: {len(X_test)}")

# 4. Extragerea de Caracteristici din Text (Vectorizare TF-IDF)
# TF-IDF transformă textul în vectori numerici.
# max_features=5000: Limitează la cele mai frecvente 5000 de cuvinte/perechi de cuvinte.
# ngram_range=(1, 2): Include cuvinte singure (unigrams) și perechi de cuvinte (bigrams).
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Antrenează (fit) vectorizatorul pe datele de antrenare și transformă (transform)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# Doar transformă datele de testare, folosind vocabularul învățat din setul de antrenare
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 5. Antrenarea Modelului de Regresie Logistică
# max_iter=1000: Crește numărul maxim de iterații pentru a asigura convergența.
# solver='liblinear': Un algoritm eficient pentru seturi de date mici/medii.
logistic_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)

# Antrenarea modelului
logistic_model.fit(X_train_tfidf, y_train)
print("\nModelul de Regresie Logistică a fost antrenat cu succes!")

# 6. Evaluarea Modelului
# Realizează predicții pe setul de testare
y_pred = logistic_model.predict(X_test_tfidf)

# Afișează Acuratețea și Raportul de Clasificare
print("\n--- Rezultate Evaluare pe Setul de Testare ---")
print(f"Acuratețe: {accuracy_score(y_test, y_pred):.4f}")
print("\nRaport de Clasificare:")
print(classification_report(y_test, y_pred, target_names=['Scris de Om (0)', 'Generat de Mașină (1)']))