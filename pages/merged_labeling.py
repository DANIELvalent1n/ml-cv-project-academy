import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import require_auth, init_session_state
from database.db_manager import DatabaseManager
from models.yahoo_news_classifier import YahooAnswersClassifier, YAHOO_CATEGORIES
from models.news_classifier import NewsClassifier

# Page config
st.set_page_config(
    page_title="NewsAI Pro - Multi Classifier",
    page_icon="📰",
    layout="wide"
)

# Initialize session state
init_session_state()

@require_auth
def main():
    st.title("📰 NewsAI Pro - Multi Classifier")
    st.markdown("Classify Yahoo Answers topics and news articles in one interface.")

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
    tab_yahoo, tab_news = st.tabs(["❓ Yahoo! Topic Classification", "🏷️ News Classification"])

    # ----------------- Yahoo Tab -----------------
    with tab_yahoo:
        st.markdown("### Enter Question Content")
        col1, col2 = st.columns([2,1])

        with col1:
            text_input = st.text_area(
                "Question Title, Content, and Best Answer (Combined)",
                height=300,
                placeholder="Example: Why is the sky blue? ..."
            )
        
        with col2:
            st.markdown("### 📚 10 Topics")
            st.markdown(", ".join(f"**{cat}**" for cat in YAHOO_CATEGORIES))
            if not yahoo_classifier.supports_proba:
                st.warning("⚠️ LinearSVC model may not provide confidence scores.")

        if st.button("🔍 Classify Yahoo Topic", key="yahoo_classify"):
            if text_input and len(text_input.strip()) > 50:
                with st.spinner("Analyzing Yahoo topic..."):
                    try:
                        prediction, confidence, scores = yahoo_classifier.predict(text_input)
                        db.save_classification(st.session_state.user_id, text_input, f"Yahoo: {prediction}", confidence)

                        st.markdown("### 🎯 Topic Classification Results")
                        col1, col2, col3 = st.columns([2,1,1])

                        with col1:
                            st.markdown(f"<div class='result-badge'>{prediction}</div>", unsafe_allow_html=True)
                        
                        conf_class = "confidence-high" if confidence > 0.7 else "confidence-low" if confidence == 0.0 else "confidence-medium"
                        with col2:
                            st.markdown(f"<p class='{conf_class}'>{confidence*100:.1f}%</p>", unsafe_allow_html=True)
                        with col3:
                            if confidence > 0.7:
                                st.success("✅ High Confidence")
                            elif confidence > 0.5:
                                st.warning("⚠️ Medium Confidence")
                            else:
                                st.error("❌ Low Confidence")

                        # Probability distribution chart
                        df_scores = pd.DataFrame({'Topic': list(scores.keys()), 'Score (%)': [v*100 for v in scores.values()]})
                        fig = go.Figure([go.Bar(x=df_scores['Topic'], y=df_scores['Score (%)'], text=[f"{v:.1f}%" for v in df_scores['Score (%)']], textposition='auto')])
                        fig.update_layout(xaxis_title="Topic", yaxis_title="Score (%)", yaxis_range=[0,100])
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Enter sufficient text for classification (>50 chars).")

    # ----------------- News Tab -----------------
    with tab_news:
        st.markdown("### Enter News Article")
        col1, col2 = st.columns([2,1])

        with col1:
            news_text = st.text_area("News Article", height=250)
        
        with col2:
            st.markdown("### 📊 Categories")
            st.info("**World**, **Sports**, **Business**, **Sci/Tech**")

        if st.button("🔍 Classify News Article", key="news_classify"):
            if news_text and len(news_text.strip()) > 20:
                with st.spinner("Classifying news..."):
                    category, confidence, scores = news_classifier.predict(news_text)
                    db.save_classification(st.session_state.user_id, news_text, category, confidence)

                    st.markdown("### 🎯 Classification Results")
                    col1, col2, col3 = st.columns([2,1,1])
                    with col1:
                        st.markdown(f"<span class='category-badge'>{category}</span>", unsafe_allow_html=True)
                    conf_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.5 else "confidence-low"
                    with col2:
                        st.markdown(f"<p class='{conf_class}'>{confidence*100:.1f}%</p>", unsafe_allow_html=True)
                    with col3:
                        if confidence > 0.7:
                            st.success("✅ High Confidence")
                        elif confidence > 0.5:
                            st.warning("⚠️ Medium Confidence")
                        else:
                            st.error("❌ Low Confidence")

                    # Probability distribution chart
                    fig = go.Figure([go.Bar(x=list(scores.keys()), y=[v*100 for v in scores.values()], text=[f"{v*100:.1f}%" for v in scores.values()], textposition='auto')])
                    fig.update_layout(xaxis_title="Category", yaxis_title="Probability (%)", yaxis_range=[0,100])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please enter a news article with at least 20 characters")

if __name__ == "__main__":
    main()
