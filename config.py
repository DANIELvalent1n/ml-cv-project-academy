import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "trained_models"
DATABASE_DIR = BASE_DIR / "database"
ASSETS_DIR = BASE_DIR / "assets"

# Create directories
for dir_path in [DATA_DIR, DATASETS_DIR, MODELS_DIR, DATABASE_DIR, ASSETS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Database
DB_PATH = DATABASE_DIR / "news.db"

# Model configs
NEWS_CATEGORIES = [ "World", "Sports", "Business", "Sci/Tech"]
FAKE_NEWS_THRESHOLD = 0.5
AI_IMAGE_THRESHOLD = 0.6

# Hugging Face models
CLASSIFICATION_MODEL = "distilbert-base-uncased"
FAKE_NEWS_MODEL = "distilbert-base-uncased"
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
TEXT_GENERATION_MODEL = "gpt2"

# App settings
APP_TITLE = "NewsAI Pro"
APP_ICON = "📰"
SESSION_TIMEOUT = 3600  # 1 hour

# Security
SALT_ROUNDS = 12