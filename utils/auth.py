import streamlit as st
from database.db_manager import DatabaseManager
import re

db = DatabaseManager()

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    return True, "Valid"

def login_user(username, password):
    """Login user"""
    success, user = db.verify_user(username, password)
    if success:
        st.session_state.logged_in = True
        st.session_state.user = user
        st.session_state.user_id = user['id']
        return True, "Login successful!"
    return False, "Invalid username or password"

def register_user(username, email, password, confirm_password):
    """Register new user"""
    # Validation
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if password != confirm_password:
        return False, "Passwords do not match"
    
    valid_pass, msg = is_valid_password(password)
    if not valid_pass:
        return False, msg
    
    # Create user
    success, result = db.create_user(username, email, password)
    if success:
        return True, "Account created successfully! Please login."
    return False, result

def logout_user():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.user_id = None

def require_auth(func):
    """Decorator to require authentication"""
    def wrapper(*args, **kwargs):
        init_session_state()
        if not st.session_state.logged_in:
            st.warning("Please login to access this page")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

def show_login_form():
    """Display login form"""
    st.title("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login", use_container_width=True)
        
        if submit:
            if username and password:
                success, message = login_user(username, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please fill in all fields")

def show_register_form():
    """Display registration form"""
    st.title("📝 Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register", use_container_width=True)
        
        if submit:
            if username and email and password and confirm_password:
                success, message = register_user(username, email, password, confirm_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("Please fill in all fields")