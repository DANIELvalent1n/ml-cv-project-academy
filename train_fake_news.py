# Step 1: Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import config
import pickle

# Step 2: Load both CSVs
fake_df = pd.read_csv('data/datasets/Fake.csv')
true_df = pd.read_csv('data/datasets/True.csv')

# Step 3: Add labels
fake_df['label'] = 1  # 1 = FAKE
true_df['label'] = 0  # 0 = TRUE

# Step 4: Combine datasets
df = pd.concat([fake_df, true_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Optional: Drop date if not useful
df = df.drop(columns=['date'], errors='ignore')

# Step 5: Combine title + text into one column
df['content'] = df['title'] + " " + df['text']

# Step 6: Split into features and labels
X = df['content']
y = df['label']

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Text vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1,2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 9: Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 10: Predictions & Evaluation
y_pred = model.predict(X_test_tfidf)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Save the model
print("\n6. Salvare model...")
model_path = config.MODELS_DIR / "kaggle_fake_news_detector.pkl"
vectorizer_path = config.MODELS_DIR / "kaggle_fake_news_vectorizer.pkl"

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
with open(vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"   ✓ Model salvat: {model_path}")
print(f"   ✓ Vectorizer salvat: {vectorizer_path}")

# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
