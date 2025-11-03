import sqlite3
import bcrypt
from datetime import datetime
from pathlib import Path
import json
import config

class DatabaseManager:
    def __init__(self):
        self.db_path = config.DB_PATH
        self.init_database()
    
    def get_connection(self):
        """Create database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # News classifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                news_text TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Fake news detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fake_news_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                news_text TEXT NOT NULL,
                news_url TEXT,
                image_path TEXT,
                fake_probability REAL,
                ai_image_probability REAL,
                verdict TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Generated news table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_path TEXT NOT NULL,
                image_caption TEXT,
                generated_title TEXT,
                generated_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # User management
    def create_user(self, username, email, password):
        """Create new user"""
        try:
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return True, user_id
        except sqlite3.IntegrityError:
            return False, "Username or email already exists"
    
    def verify_user(self, username, password):
        """Verify user credentials"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
            self.update_last_login(user['id'])
            return True, dict(user)
        return False, None
    
    def update_last_login(self, user_id):
        """Update last login timestamp"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE users SET last_login = ? WHERE id = ?',
            (datetime.now(), user_id)
        )
        conn.commit()
        conn.close()
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        conn.close()
        return dict(user) if user else None
    
    # News classification methods
    def save_classification(self, user_id, news_text, category, confidence):
        """Save news classification result"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO news_classifications 
               (user_id, news_text, predicted_category, confidence) 
               VALUES (?, ?, ?, ?)''',
            (user_id, news_text, category, confidence)
        )
        conn.commit()
        conn.close()
    
    def get_user_classifications(self, user_id, limit=10):
        """Get user's classification history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT * FROM news_classifications 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT ?''',
            (user_id, limit)
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    # Fake news detection methods
    def save_fake_detection(self, user_id, news_text, news_url, image_path, 
                           fake_prob, ai_image_prob, verdict):
        """Save fake news detection result"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO fake_news_detections 
               (user_id, news_text, news_url, image_path, fake_probability, 
                ai_image_probability, verdict) 
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (user_id, news_text, news_url, image_path, fake_prob, ai_image_prob, verdict)
        )
        conn.commit()
        conn.close()
    
    def get_user_detections(self, user_id, limit=10):
        """Get user's detection history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT * FROM fake_news_detections 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT ?''',
            (user_id, limit)
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    # Generated news methods
    def save_generated_news(self, user_id, image_path, caption, title, content):
        """Save generated news"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO generated_news 
               (user_id, image_path, image_caption, generated_title, generated_content) 
               VALUES (?, ?, ?, ?, ?)''',
            (user_id, image_path, caption, title, content)
        )
        conn.commit()
        conn.close()
    
    def get_user_generated_news(self, user_id, limit=10):
        """Get user's generated news history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT * FROM generated_news 
               WHERE user_id = ? 
               ORDER BY created_at DESC 
               LIMIT ?''',
            (user_id, limit)
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    # Statistics
    def get_user_stats(self, user_id):
        """Get user statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total classifications
        cursor.execute('SELECT COUNT(*) FROM news_classifications WHERE user_id = ?', (user_id,))
        stats['total_classifications'] = cursor.fetchone()[0]
        
        # Total detections
        cursor.execute('SELECT COUNT(*) FROM fake_news_detections WHERE user_id = ?', (user_id,))
        stats['total_detections'] = cursor.fetchone()[0]
        
        # Total generated news
        cursor.execute('SELECT COUNT(*) FROM generated_news WHERE user_id = ?', (user_id,))
        stats['total_generated'] = cursor.fetchone()[0]
        
        # Category distribution
        cursor.execute(
            '''SELECT predicted_category, COUNT(*) as count 
               FROM news_classifications 
               WHERE user_id = ? 
               GROUP BY predicted_category''',
            (user_id,)
        )
        stats['category_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        return stats