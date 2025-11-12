import config
from utils.auth import require_auth, init_session_state, show_login_form, show_register_form, logout_user
import streamlit as st

class LoginModel:
    def __init__(self):
        pass

    def login():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"<h1 style='text-align: center;'>{config.APP_ICON} {config.APP_TITLE}</h1>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.info("Please login or register to continue")
            
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
            
            with tab1:
                show_login_form()
            
            with tab2:
                show_register_form()