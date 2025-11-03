import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import config
import streamlit as st

class AIImageDetector:
    def __init__(self):
        self.model_path_cnn = config.MODELS_DIR / "ai_image_detector_cnn.pth"
        self.model = None
        self.model_type = None
        self.device = None
        self.transform = None
        
        # CNN-only calibration settings
        self.temperature = 3.0      # Temperature scaling (higher = less confident)
        self.bias_correction = -0.50  # Shift predictions away from "always AI"
        self.threshold = 0.65       # Conservative threshold
        
        self.load_or_train()
    
    def load_or_train(self):
        """Load existing model"""
        # Try to load CNN model first
        if self.model_path_cnn.exists():
            try:
                self.load_cnn_model()
                self.model_type = 'cnn'
                print(f"✅ CNN model loaded (Temperature: {self.temperature}, Bias: {self.bias_correction})")
                return
            except Exception as e:
                print(f"⚠️ Could not load CNN model: {e}")
    
    def load_cnn_model(self):
        """Load trained CNN model from .pth file"""
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        
        # Define the same CNN architecture used in training
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(128 * 16 * 16, 512)
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
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path_cnn, map_location=self.device))
        self.model.eval()  # Critical: eval mode
        
        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """Preprocess image to ensure correct format"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def calibrate_probability(self, raw_prob):
        """
        Apply temperature scaling + bias correction to fix overconfidence
        
        Temperature scaling: Makes predictions less extreme
        Bias correction: Shifts predictions away from bias
        """
        import math
        
        # Clip to avoid log(0)
        epsilon = 1e-7
        raw_prob = max(epsilon, min(1 - epsilon, raw_prob))
        
        # Convert to logit
        logit = math.log(raw_prob / (1 - raw_prob))
        
        # Apply temperature (reduces confidence)
        calibrated_logit = logit / self.temperature
        
        # Apply bias correction (shifts distribution)
        calibrated_logit += self.bias_correction
        
        # Convert back to probability
        calibrated_prob = 1 / (1 + math.exp(-calibrated_logit))
        
        # Ensure in valid range
        calibrated_prob = max(0.0, min(1.0, calibrated_prob))
        
        return calibrated_prob
    
    def predict_with_cnn(self, image):
        """Predict using CNN with calibration"""
        import torch
        
        # Preprocess
        image = self.preprocess_image(image)
        
        # Transform
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            raw_ai_prob = probabilities[0][1].item()
            raw_real_prob = probabilities[0][0].item()
        
        # Apply calibration
        calibrated_ai_prob = self.calibrate_probability(raw_ai_prob)
        calibrated_real_prob = 1 - calibrated_ai_prob
        
        # Determine confidence based on distance from threshold
        distance_from_threshold = abs(calibrated_ai_prob - 0.5)
        if distance_from_threshold > 0.3:
            confidence = 'High'
        elif distance_from_threshold > 0.15:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return {
            'is_ai_generated': calibrated_ai_prob > self.threshold,
            'ai_probability': float(calibrated_ai_prob),
            'real_probability': float(calibrated_real_prob),
            'confidence': confidence,
            'model_type': 'CNN (Calibrated)',
            'raw_prediction': float(raw_ai_prob),  # For debugging
        }
    
    def predict(self, image):
        """Main prediction method - CNN only"""
        image = self.preprocess_image(image)
        
        if self.model_type == 'cnn':
            try:
                return self.predict_with_cnn(image)
            except Exception as e:
                print(f"⚠️ CNN prediction failed: {e}")
    
    def analyze_image(self, image):
        """Comprehensive image analysis"""
        result = self.predict(image)
        
        # Extract image features for analysis notes
        image = self.preprocess_image(image)
        
        # Determine verdict with calibrated threshold
        ai_prob = result['ai_probability']
        
        if ai_prob > 0.75:
            verdict = "Likely AI-Generated"
        elif ai_prob > 0.45:
            verdict = "Uncertain"
        else:
            verdict = "Likely Real Photo"
        
        result['verdict'] = verdict
        
        # Add detailed analysis notes
        notes = []
        
        # Model info
        if result['model_type'] == 'CNN (Calibrated)':
            notes.append(f"CNN Model (Temperature scaling: {self.temperature})")
            if 'raw_prediction' in result:
                raw = result['raw_prediction']
                calib = result['ai_probability']
                notes.append(f"Raw prediction: {raw*100:.1f}% → Calibrated: {calib*100:.1f}%")
        
        # Confidence explanation
        if result['confidence'] == 'Low':
            notes.append("Low confidence - image characteristics are ambiguous")
        
        if not notes:
            notes.append("Standard image characteristics detected")
        
        result['analysis_notes'] = notes
        
        return result