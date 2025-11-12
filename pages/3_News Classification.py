import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import config

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import require_auth, init_session_state
from models.login import LoginModel
from database.db_manager import DatabaseManager
from models.yahoo_news_classifier import YahooAnswersClassifier, YAHOO_CATEGORIES
from models.news_classifier import NewsClassifier

# Page config
st.set_page_config(
    page_title="📰 NewsAI Pro - Multi Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
.result-badge, .category-badge {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 12px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    color: white;
    font-weight: bold;
    font-size: 1.2rem;
    text-align: center;
}
.confidence-high {
    color: #16a34a;
    font-weight: bold;
}
.confidence-medium {
    color: #eab308;
    font-weight: bold;
}
.confidence-low {
    color: #dc2626;
    font-weight: bold;
}
.stProgress > div > div > div {
    background-color: #3b82f6;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
init_session_state()

def extract_text_from_url(url):
    """Fetch article content from a URL."""
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        article_text = " ".join(paragraphs)
        return article_text[:5000]  # limit length
    except Exception as e:
        st.error(f"Failed to extract article: {e}")
        return None


def display_results(title, prediction, confidence, scores):
    """Reusable results UI block."""
    st.markdown(f"### 🎯 {title} Results")
    st.markdown(f"<div class='result-badge'>{prediction}</div>", unsafe_allow_html=True)
    st.progress(confidence)
    st.markdown(f"**Confidence:** {confidence*100:.1f}%")
    if confidence > 0.7:
        st.success("✅ High Confidence")
    elif confidence > 0.5:
        st.warning("⚠️ Medium Confidence")
    else:
        st.error("❌ Low Confidence")

    # Probability distribution chart
    df_scores = pd.DataFrame({'Label': list(scores.keys()), 'Score (%)': [v*100 for v in scores.values()]})
    fig = go.Figure([
        go.Bar(
            x=df_scores['Label'], 
            y=df_scores['Score (%)'], 
            text=[f"{v:.1f}%" for v in df_scores['Score (%)']], 
            textposition='auto'
        )
    ])
    fig.update_layout(xaxis_title="Category", yaxis_title="Score (%)", yaxis_range=[0,100])
    st.plotly_chart(fig, use_container_width=True)


def main():
    if st.session_state.logged_in:

        st.title("🧠 NewsAI Pro - News Classifier")
        st.markdown("Easily classify **News Articles** using AI-powered models.")

        # Initialize models
        if 'yahoo_classifier' not in st.session_state:
            with st.spinner("Loading Yahoo Answers model..."):
                st.session_state.yahoo_classifier = YahooAnswersClassifier()
        if 'news_classifier' not in st.session_state:
            with st.spinner("Loading News classifier model..."):
                st.session_state.news_classifier = NewsClassifier()
        
        yahoo_classifier = st.session_state.yahoo_classifier
        news_classifier = st.session_state.news_classifier
        db = DatabaseManager()

        # Tabs
        tab_yahoo, tab_news, tab_url = st.tabs(["❓ Yahoo! Topics Classification", "🏷️ AG News Classification", "🌐 URL Classification"])

        # ----------------- Yahoo Tab -----------------
        with tab_yahoo:
            st.subheader("Classify Yahoo! Topics")
            text_input = st.text_area("Write/Paste the article below:", height=250, placeholder="Paste or type the article text here...")
            st.caption("⚡ Tip: More text gives better results.")

            if st.button("🔍 Classify based on Yahoo Topics dataset"):
                if text_input and len(text_input.strip()) > 50:
                    with st.spinner("Analyzing Yahoo topic..."):
                        try:
                            prediction, confidence, scores = yahoo_classifier.predict(text_input)
                            db.save_classification(st.session_state.user_id, text_input, f"Yahoo: {prediction}", confidence)
                            display_results("Yahoo Topic Classification", prediction, confidence, scores)
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please enter at least 50 characters for better accuracy.")

        # ----------------- News Tab -----------------
        with tab_news:
            st.subheader("Classify AG News Article")
            news_text = st.text_area("Write/Paste the article below", height=250, placeholder="Paste or type the article text here...")

            if st.button("🔍 Classify News Article"):
                if news_text and len(news_text.strip()) > 20:
                    with st.spinner("Classifying news..."):
                        category, confidence, scores = news_classifier.predict(news_text)
                        db.save_classification(st.session_state.user_id, news_text, category, confidence)
                        display_results("News Classification", category, confidence, scores)
                else:
                    st.warning("Please enter at least 20 characters.")

        # ----------------- URL Tab -----------------
        with tab_url:
            st.subheader("🌐 Classify news from a Web URL")
            url_input = st.text_input("Enter Article URL", placeholder="https://example.com/news-article")
            
            if st.button("🔍 Analyze URL"):
                if url_input:
                    with st.spinner("Extracting and classifying article..."):
                        article_text = extract_text_from_url(url_input)
                        if article_text:
                            st.info("✅ Article successfully extracted. Now classifying with both models...")

                            # Yahoo classification
                            yahoo_pred, yahoo_conf, yahoo_scores = yahoo_classifier.predict(article_text)
                            display_results("Yahoo Topic Classification", yahoo_pred, yahoo_conf, yahoo_scores)

                            # News classification
                            news_pred, news_conf, news_scores = news_classifier.predict(article_text)
                            display_results("News Classification", news_pred, news_conf, news_scores)

                            db.save_classification(st.session_state.user_id, url_input, f"Yahoo: {yahoo_pred} / News: {news_pred}", (yahoo_conf + news_conf) / 2)
                else:
                    st.warning("Please enter a valid URL.")
    else:
        LoginModel.login()


if __name__ == "__main__":
    main()
