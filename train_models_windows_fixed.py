"""
Script pentru antrenarea modelelor pe dataset-uri reale - VERSIUNE WINDOWS FIXED
Rulează: python train_models_windows_fixed.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import config
import csv

def train_news_classifier():
    """
    Antrenează clasificatorul de știri pe AG News Dataset
    """
    print("=" * 60)
    print("ANTRENARE NEWS CLASSIFIER")
    print("=" * 60)
    
    try:
        # Încarcă dataset
        print("\n1. Încărcare dataset AG News...")
        from datasets import load_dataset
        
        # Descarcă și încarcă dataset
        dataset = load_dataset("ag_news")
        
        # Extrage datele
        train_data = dataset['train']
        test_data = dataset['test']
        
        print(f"   ✓ Training samples: {len(train_data)}")
        print(f"   ✓ Test samples: {len(test_data)}")
        
        # Pregătește datele
        print("\n2. Pregătire date...")
        X_train = [item['text'] for item in train_data]
        y_train = [item['label'] for item in train_data]
        
        X_test = [item['text'] for item in test_data]
        y_test = [item['label'] for item in test_data]
        
        # Mapare labels la categorii
        label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
        y_train = [label_map[label] for label in y_train]
        y_test = [label_map[label] for label in y_test]
        
        # Creare vectorizer
        print("\n3. Creare TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        print(f"   ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # Antrenare model
        print("\n4. Antrenare model Logistic Regression...")
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train_vec, y_train)
        
        # Evaluare
        print("\n5. Evaluare model...")
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n   ✓ Accuracy: {accuracy*100:.2f}%")
        print("\n   Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Salvare model
        print("\n6. Salvare model...")
        model_path = config.MODELS_DIR / "news_classifier.pkl"
        vectorizer_path = config.MODELS_DIR / "news_vectorizer.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print(f"   ✓ Model salvat: {model_path}")
        print(f"   ✓ Vectorizer salvat: {vectorizer_path}")
        
        print("\n✅ ANTRENARE NEWS CLASSIFIER COMPLETĂ!")
        return True
        
    except Exception as e:
        print(f"\n❌ EROARE: {e}")
        print("\nAsigură-te că ai instalat: pip install datasets")
        return False

def train_fake_news_detector():
    """
    Antrenează detectorul de fake news pe dataset real
    """
    print("\n" + "=" * 60)
    print("ANTRENARE FAKE NEWS DETECTOR")
    print("=" * 60)
    
    try:
        # Încarcă dataset
        print("\n1. Încărcare dataset...")
        
        # Opțiunea 1: Kaggle Fake News Dataset
        liar_dir = config.DATASETS_DIR / "liar"
        
        if liar_dir.exists():
            print(f"   Încărcare LIAR dataset din: {liar_dir}")
            # Columne conform README LIAR
            colnames = [
                "id", "label", "statement", "subjects", "speaker", "job_title",
                "state", "party", "barely_true_count", "false_count", "half_true_count",
                "mostly_true_count", "pants_on_fire_count", "context"
            ]
            
            # Preferăm train.tsv dacă există, altfel concatenăm toate .tsv din folder
            train_file = liar_dir / "train.tsv"
            valid_file = liar_dir / "valid.tsv"
            test_file  = liar_dir / "test.tsv"
            
            if train_file.exists():
                df = pd.read_csv(train_file, sep="\t", header=None, names=colnames,
                                 quoting=csv.QUOTE_NONE, encoding="utf-8", on_bad_lines="skip")
                # optional: concat cu valid/test dacă există
                parts = [df]
                if valid_file.exists():
                    parts.append(pd.read_csv(valid_file, sep="\t", header=None, names=colnames,
                                             quoting=csv.QUOTE_NONE, encoding="utf-8", on_bad_lines="skip"))
                if test_file.exists():
                    parts.append(pd.read_csv(test_file, sep="\t", header=None, names=colnames,
                                             quoting=csv.QUOTE_NONE, encoding="utf-8", on_bad_lines="skip"))
                df = pd.concat(parts, ignore_index=True)
            else:
                tsvs = sorted(liar_dir.glob("*.tsv"))
                if not tsvs:
                    raise FileNotFoundError(f"No .tsv files found in {liar_dir}")
                dfs = [pd.read_csv(p, sep="\t", header=None, names=colnames,
                                   quoting=csv.QUOTE_NONE, encoding="utf-8", on_bad_lines="skip")
                       for p in tsvs]
                df = pd.concat(dfs, ignore_index=True)
            
            # Curățare și mapare la binar:
            df = df.dropna(subset=['statement', 'label'])
            # Labels în LIAR sunt stringuri: 'true','mostly-true','half-true', 'false', 'pants-fire',...
            # Considerăm reale doar 'true' și 'mostly-true'
            df['label_norm'] = df['label'].astype(str).str.lower().str.strip()
            df['label_bin'] = df['label_norm'].apply(lambda x: 0 if x in ("true", "mostly-true") else 1)
            
            texts = df['statement'].astype(str).tolist()
            labels = df['label_bin'].tolist()
        
        else:
            raise FileNotFoundError("Nu am găsit nici dataset Kaggle (data/datasets/fake_news/train.csv) nici folderul datasets/liar")
        
        print(f"   ✓ Total samples: {len(texts)}")
        print(f"   ✓ Fake: {sum(labels)}, Real: {len(labels) - sum(labels)}")
        
        # Split data
        print("\n2. Split train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Creare vectorizer
        print("\n3. Creare TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Antrenare model
        print("\n4. Antrenare Random Forest...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train_vec, y_train)
        
        # Evaluare
        print("\n5. Evaluare model...")
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n   ✓ Accuracy: {accuracy*100:.2f}%")
        print("\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Salvare model
        print("\n6. Salvare model...")
        model_path = config.MODELS_DIR / "fake_news_detector.pkl"
        vectorizer_path = config.MODELS_DIR / "fake_news_vectorizer.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print(f"   ✓ Model salvat: {model_path}")
        print(f"   ✓ Vectorizer salvat: {vectorizer_path}")
        
        print("\n✅ ANTRENARE FAKE NEWS DETECTOR COMPLETĂ!")
        return True
        
    except Exception as e:
        print(f"\n❌ EROARE: {e}")
        print("\nAsigură-te că ai:")
        print("  - Dataset Kaggle în: data/datasets/fake_news/train.csv")
        print("  SAU")
        print("  - Folder local LIAR în: data/datasets/liar/ (train.tsv, valid.tsv, test.tsv sau .tsv files)")
        return False
# ...existing code...

def train_ai_image_detector():
    """
    Antrenează detectorul de imagini AI folosind CNN - WINDOWS COMPATIBLE
    """
    print("\n" + "=" * 60)
    print("ANTRENARE AI IMAGE DETECTOR")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader, TensorDataset
        from PIL import Image
        import os
        
        print("\n1. Verificare GPU...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   ✓ Device: {device}")
        
        # Transformări
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Reduced size for faster training
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Încarcă dataset
        print("\n2. Încărcare imagini...")
        real_dir = config.DATASETS_DIR / "cifake" / "real"
        fake_dir = config.DATASETS_DIR / "cifake" / "fake"
        
        if not real_dir.exists() or not fake_dir.exists():
            print(f"\n❌ Dataset CIFAKE nu a fost găsit!")
            print(f"   Descarcă de pe: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
            print(f"   Extrage în: {config.DATASETS_DIR / 'cifake'}")
            print(f"   Structura trebuie să fie:")
            print(f"   {config.DATASETS_DIR / 'cifake' / 'real'}")
            print(f"   {config.DATASETS_DIR / 'cifake' / 'fake'}")
            return False
        
        # Load images - LIMITED to 5000 for faster training
        print("   Încărcare imagini reale...")
        real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5000]
        
        print("   Încărcare imagini AI...")
        fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5000]
        
        print(f"   ✓ Real images: {len(real_images)}")
        print(f"   ✓ Fake images: {len(fake_images)}")
        
        # Load and preprocess all images into memory (Windows-compatible)
        print("\n3. Procesare imagini în memorie...")
        X_data = []
        y_data = []
        
        # Process real images
        for i, img_path in enumerate(real_images):
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                X_data.append(image_tensor)
                y_data.append(0)  # 0 = real
                
                if (i + 1) % 500 == 0:
                    print(f"   Real: {i + 1}/{len(real_images)}")
            except Exception as e:
                continue
        
        # Process fake images
        for i, img_path in enumerate(fake_images):
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                X_data.append(image_tensor)
                y_data.append(1)  # 1 = fake
                
                if (i + 1) % 500 == 0:
                    print(f"   Fake: {i + 1}/{len(fake_images)}")
            except Exception as e:
                continue
        
        # Convert to tensors
        X_data = torch.stack(X_data)
        y_data = torch.tensor(y_data, dtype=torch.long)
        
        print(f"\n   ✓ Total processed: {len(X_data)}")
        
        # Split
        train_size = int(0.8 * len(X_data))
        indices = torch.randperm(len(X_data))
        
        X_train = X_data[indices[:train_size]]
        y_train = y_data[indices[:train_size]]
        X_test = X_data[indices[train_size:]]
        y_test = y_data[indices[train_size:]]
        
        print(f"   ✓ Training samples: {len(X_train)}")
        print(f"   ✓ Test samples: {len(X_test)}")
        
        # Create TensorDatasets and DataLoaders with num_workers=0 for Windows
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Model CNN
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjusted for 128x128 input
                self.fc2 = nn.Linear(512, 2)
                self.dropout = nn.Dropout(0.5)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(-1, 128 * 16 * 16)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        print("\n4. Creare model CNN...")
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Antrenare
        print("\n5. Antrenare model...")
        num_epochs = 10
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (i + 1) % 50 == 0:
                    print(f'   Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            accuracy = 100 * correct / total
            print(f'   Epoch [{epoch+1}/{num_epochs}] - Accuracy: {accuracy:.2f}%, Loss: {running_loss/len(train_loader):.4f}')
        
        # Evaluare
        print("\n6. Evaluare model...")
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'   ✓ Test Accuracy: {accuracy:.2f}%')
        
        # Salvare model
        print("\n7. Salvare model...")
        model_path = config.MODELS_DIR / "ai_image_detector_cnn.pth"
        torch.save(model.state_dict(), model_path)
        print(f"   ✓ Model salvat: {model_path}")
        
        print("\n✅ ANTRENARE AI IMAGE DETECTOR COMPLETĂ!")
        return True
        
    except Exception as e:
        import traceback
        print(f"\n❌ EROARE: {e}")
        print("\nDetalii:")
        traceback.print_exc()
        print("\nAsigură-te că ai:")
        print("  - Dataset CIFAKE în: data/datasets/cifake/")
        print("  - torch și torchvision instalate: pip install torch torchvision")
        print("  - Suficient RAM (minim 8GB recomandat)")
        return False


def main():
    """
    Funcție principală pentru antrenarea tuturor modelelor
    """
    print("=" * 60)
    print("SISTEM DE ANTRENARE MODELE ML - WINDOWS VERSION")
    print("=" * 60)
    print("\nVERSIUNE FIXATĂ PENTRU WINDOWS:")
    print("  ✓ num_workers=0 (fix multiprocessing)")
    print("  ✓ Imagini încărcate în memorie")
    print("  ✓ Rezoluție redusă (128x128) pentru viteză")
    
    print("\n\nCe model dorești să antrenezi?")
    print("1. News Classifier (AG News)")
    print("2. Fake News Detector (Kaggle/LIAR)")
    print("3. AI Image Detector (CIFAKE) - WINDOWS FIXED")
    print("4. Toate modelele")
    print("0. Ieșire")
    
    choice = input("\nAlege opțiunea (0-4): ").strip()
    
    if choice == '1':
        train_news_classifier()
    elif choice == '2':
        train_fake_news_detector()
    elif choice == '3':
        train_ai_image_detector()
    elif choice == '4':
        print("\n🚀 Antrenare toate modelele...\n")
        train_news_classifier()
        train_fake_news_detector()
        train_ai_image_detector()
    elif choice == '0':
        print("Ieșire...")
    else:
        print("❌ Opțiune invalidă!")


if __name__ == "__main__":
    main()