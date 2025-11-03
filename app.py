import streamlit as st
from utils.auth import init_session_state, show_login_form, show_register_form, logout_user
import config

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #333;
    }
    
    .feature-desc {
        text-align: center;
        color: #666;
        line-height: 1.6;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stats-label {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

def show_homepage():
    """Display homepage"""
    # Hero section
    st.markdown("""
        <div class="hero-section">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">📰 NewsAI Pro</h1>
            <h3>Powered by Artificial Intelligence & Machine Learning</h3>
            <p style="font-size: 1.2rem; margin-top: 1rem;">
                Advanced news analysis, fake news detection, and AI-powered news generation
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("<h2 style='text-align: center; margin: 3rem 0 2rem 0;'>🚀 Powerful Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🏷️</div>
                <div class="feature-title">News Classification</div>
                <div class="feature-desc">
                    Automatically categorize news articles into World, Sports, Business, or Sci/Tech using advanced ML algorithms.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Fake News Detection</div>
                <div class="feature-desc">
                    Detect fake news with AI-powered text analysis and image verification to ensure content authenticity.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">✍️</div>
                <div class="feature-title">News Generation</div>
                <div class="feature-desc">
                    Generate complete news articles from images using computer vision and natural language processing.
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Technology section
    st.markdown("<h2 style='text-align: center; margin: 3rem 0 2rem 0;'>🛠️ Technology Stack</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="stats-card">
                <div class="stats-number">🤖</div>
                <div class="stats-label">Machine Learning</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stats-card">
                <div class="stats-number">👁️</div>
                <div class="stats-label">Computer Vision</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="stats-card">
                <div class="stats-number">📊</div>
                <div class="stats-label">NLP</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="stats-card">
                <div class="stats-number">🗄️</div>
                <div class="stats-label">SQLite</div>
            </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("<h2 style='text-align: center; margin: 3rem 0 2rem 0;'>📋 How It Works</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Step 1: Login/Register**\n\nCreate your account or login to access all features.")
    
    with col2:
        st.info("**Step 2: Choose Feature**\n\nSelect from news classification, fake news detection, or news generation.")
    
    with col3:
        st.info("**Step 3: Get Results**\n\nReceive instant AI-powered analysis and insights.")
    
    # CTA Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                <h2>Ready to Get Started?</h2>
                <p>Login or register to unlock the full power of NewsAI Pro</p>
            </div>
        """, unsafe_allow_html=True)

def main():
    """Main application"""
    # Sidebar
    with st.sidebar:
        st.markdown(f"<h1 style='text-align: center;'>{config.APP_ICON} {config.APP_TITLE}</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        if st.session_state.logged_in:
            st.success(f"👤 Welcome, {st.session_state.user['username']}!")
            st.markdown("---")
            
            # Navigation
            st.markdown("### 📍 Navigation")
            st.info("Use the sidebar menu above to navigate between features")
            
            st.markdown("---")
            
            # User actions
            if st.button("🚪 Logout", use_container_width=True):
                logout_user()
                st.rerun()
        else:
            st.info("Please login or register to continue")
            
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
            
            with tab1:
                show_login_form()
            
            with tab2:
                show_register_form()
    
    # Main content
    if st.session_state.logged_in:
        show_homepage()
    else:
        # Welcome screen for non-logged users
        st.markdown("<h1 class='main-header'>Welcome to NewsAI Pro</h1>", unsafe_allow_html=True)
        
        st.markdown(""" 
            <div class="hero-section">
                <h2>🔒 Login Required</h2>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    Please login or register using the sidebar to access all features
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show features preview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🏷️</div>
                    <div class="feature-title">News Labeling</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <div class="feature-title">Fake Detection</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="feature-card">
                    <div class="feature-icon">✍️</div>
                    <div class="feature-title">News Generator</div>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()