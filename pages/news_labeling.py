import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import require_auth, init_session_state
from models.news_classifier import NewsClassifier
from database.db_manager import DatabaseManager
import plotly.graph_objects as go
import pandas as pd

# Page config
st.set_page_config(
    page_title="News Labeling",
    page_icon="🏷️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .category-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    
    .world { background: #3498db; color: white; }
    .sports { background: #2ecc71; color: white; }
    .business { background: #e74c3c; color: white; }
    .scitech { background: #9b59b6; color: white; }
    
    .confidence-high { color: #27ae60; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize
init_session_state()

@require_auth
def main():
    st.title("🏷️ News Classification & Labeling")
    st.markdown("Automatically categorize news articles using Machine Learning")
    
    # Initialize models
    if 'classifier' not in st.session_state:
        with st.spinner("Loading classification model..."):
            st.session_state.classifier = NewsClassifier()
    
    classifier = st.session_state.classifier
    db = DatabaseManager()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Classify News", "📊 Statistics", "📜 History"])
    
    with tab1:
        st.markdown("### Enter News Article")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input methods
            input_method = st.radio(
                "Input Method",
                ["✍️ Type/Paste Text", "📄 Sample Articles"],
                horizontal=True
            )
            
            if input_method == "✍️ Type/Paste Text":
                news_text = st.text_area(
                    "News Article",
                    height=250,
                    placeholder="Paste your news article here...",
                    help="Enter the complete news article text"
                )
            else:
                samples = {
                    "World News": "The United Nations Security Council convened an emergency session today to address escalating tensions in the Middle East. Diplomatic efforts are underway to prevent further conflict.",
                    "Sports News": "The championship game ended in dramatic fashion with a last-second three-pointer. The team celebrated their victory as fans rushed onto the court in jubilation.",
                    "Business News": "Stock markets rallied today following positive earnings reports from major tech companies. The Dow Jones Industrial Average gained 300 points in early trading.",
                    "Tech News": "Researchers have developed a new quantum computing algorithm that could revolutionize cryptography. The breakthrough was published in Nature today."
                }
                
                selected_sample = st.selectbox("Choose a sample article", list(samples.keys()))
                news_text = st.text_area(
                    "News Article",
                    value=samples[selected_sample],
                    height=250
                )
        
        with col2:
            st.markdown("### 📊 Categories")
            st.info("**World** - International news, politics, global events")
            st.success("**Sports** - Sports events, games, athletes")
            st.error("**Business** - Economy, markets, companies")
            st.warning("**Sci/Tech** - Science, technology, research")
        
        # Classify button
        if st.button("🔍 Classify Article", type="primary", use_container_width=True):
            if news_text and len(news_text.strip()) > 20:
                with st.spinner("Analyzing article..."):
                    # Predict
                    category, confidence, scores = classifier.predict(news_text)
                    
                    # Save to database
                    db.save_classification(
                        st.session_state.user_id,
                        news_text,
                        category,
                        confidence
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 🎯 Classification Results")
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        # Category badge
                        category_class = category.lower().replace("/", "")
                        st.markdown(f"""
                            <div style="text-align: center;">
                                <h3>Predicted Category</h3>
                                <span class="category-badge {category_class}">{category}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Confidence
                        conf_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.5 else "confidence-low"
                        st.markdown(f"""
                            <div style="text-align: center;">
                                <h3>Confidence</h3>
                                <p class="{conf_class}" style="font-size: 2rem;">{confidence*100:.1f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Status
                        if confidence > 0.7:
                            st.success("✅ High Confidence")
                        elif confidence > 0.5:
                            st.warning("⚠️ Medium Confidence")
                        else:
                            st.error("❌ Low Confidence")
                    
                    # Probability distribution
                    st.markdown("### 📊 Probability Distribution")
                    
                    # Create bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(scores.keys()),
                            y=[v*100 for v in scores.values()],
                            marker_color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
                            text=[f"{v*100:.1f}%" for v in scores.values()],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        xaxis_title="Category",
                        yaxis_title="Probability (%)",
                        yaxis_range=[0, 100],
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Analysis
                    st.markdown("### 💡 Analysis")
                    if confidence > 0.7:
                        st.info(f"The article strongly matches the **{category}** category with high confidence. The content clearly indicates this classification.")
                    elif confidence > 0.5:
                        st.warning(f"The article appears to be **{category}**, but with moderate confidence. There may be elements of other categories present.")
                    else:
                        st.error(f"The classification suggests **{category}**, but confidence is low. The article may contain mixed topics or be difficult to categorize.")
                    
            else:
                st.error("Please enter a news article with at least 20 characters")
    
    with tab2:
        st.markdown("### 📊 Classification Statistics")
        
        stats = db.get_user_stats(st.session_state.user_id)
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Classifications", stats['total_classifications'])
        
        with col2:
            if stats['category_distribution']:
                most_common = max(stats['category_distribution'], key=stats['category_distribution'].get)
                st.metric("Most Common Category", most_common)
            else:
                st.metric("Most Common Category", "N/A")
        
        with col3:
            if stats['total_classifications'] > 0:
                recent = db.get_user_classifications(st.session_state.user_id, limit=10)
                avg_conf = sum(r['confidence'] for r in recent) / len(recent)
                st.metric("Average Confidence", f"{avg_conf*100:.1f}%")
            else:
                st.metric("Average Confidence", "N/A")
        
        # Category distribution
        if stats['category_distribution']:
            st.markdown("### 📈 Category Distribution")
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(stats['category_distribution'].keys()),
                    values=list(stats['category_distribution'].values()),
                    marker_colors=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
                    hole=0.4
                )
            ])
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No classification data yet. Start classifying articles to see statistics!")
    
    with tab3:
        st.markdown("### 📜 Classification History")
        
        history = db.get_user_classifications(st.session_state.user_id, limit=20)
        
        if history:
            for i, record in enumerate(history):
                with st.expander(f"Classification #{len(history)-i} - {record['predicted_category']} ({record['created_at']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Text Preview:**")
                        st.text(record['news_text'][:200] + "..." if len(record['news_text']) > 200 else record['news_text'])
                    
                    with col2:
                        st.markdown(f"**Category:** {record['predicted_category']}")
                        st.markdown(f"**Confidence:** {record['confidence']*100:.1f}%")
                        st.markdown(f"**Date:** {record['created_at']}")
        else:
            st.info("No classification history yet. Start classifying articles to build your history!")

if __name__ == "__main__":
    main()