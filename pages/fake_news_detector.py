import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import require_auth, init_session_state
from models.fake_news_detector import FakeNewsDetector
from models.ai_image_detector import AIImageDetector
from database.db_manager import DatabaseManager
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .verdict-real {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .verdict-fake {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .verdict-suspicious {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
init_session_state()

@require_auth
def main():
    st.title("🔍 Fake News Detection System")
    st.markdown("Detect fake news using AI-powered text analysis and image verification")
    
    # Initialize models
    if 'fake_detector' not in st.session_state:
        with st.spinner("Loading fake news detection model..."):
            st.session_state.fake_detector = FakeNewsDetector()
    
    if 'image_detector' not in st.session_state:
        with st.spinner("Loading image detection model..."):
            st.session_state.image_detector = AIImageDetector()
    
    fake_detector = st.session_state.fake_detector
    image_detector = st.session_state.image_detector
    db = DatabaseManager()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Detect Fake News", "📊 Statistics", "📜 History"])
    
    with tab1:
        st.markdown("### Analyze News Credibility")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            news_text = st.text_area(
                "News Article Text",
                height=200,
                placeholder="Paste the news article text here...",
                help="Enter the complete news article text for analysis"
            )
            
            news_url = st.text_input(
                "News URL (optional)",
                placeholder="https://example.com/news-article",
                help="URL of the news article"
            )
        
        with col2:
            st.markdown("### 📸 Image Analysis")
            st.info("Upload an image from the article to check if it's AI-generated")
            
            uploaded_image = st.file_uploader(
                "Upload Image (optional)",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image from the news article"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Sample articles
        with st.expander("📄 Use Sample Articles for Testing"):
            sample_type = st.radio(
                "Select Sample Type",
                ["Real News", "Fake News"],
                horizontal=True
            )
            
            if sample_type == "Real News":
                samples = {
                    "Economic Report": "The Federal Reserve announced today that interest rates will remain unchanged following their monthly meeting. The decision comes as inflation shows signs of stabilization, according to recent economic data. Economists suggest this measured approach reflects confidence in current monetary policy effectiveness.",
                    "Science Discovery": "Researchers at MIT have published findings in the journal Nature describing a new method for carbon capture. The peer-reviewed study demonstrates a 30% improvement in efficiency over existing technologies. The research team collaborated with international partners over three years to validate their results.",
                }
            else:
                samples = {
                    "Sensational Claim": "SHOCKING! Scientists REVEAL incredible secret that CHANGES EVERYTHING! You won't BELIEVE what they found!!! This ONE weird trick will BLOW YOUR MIND! Share before it gets DELETED!!!",
                    "Conspiracy Theory": "BREAKING NEWS!!! The government is HIDING the truth about everything! Insider sources confirm UNBELIEVABLE conspiracy! They don't want you to know this AMAZING fact! Wake up people!!!",
                }
            
            selected_sample = st.selectbox("Choose sample", list(samples.keys()))
            if st.button("Load Sample"):
                news_text = samples[selected_sample]
                st.rerun()
        
        # Analyze button
        st.markdown("---")
        if st.button("🔍 Analyze News Credibility", type="primary", use_container_width=True):
            if news_text and len(news_text.strip()) > 20:
                with st.spinner("Analyzing content..."):
                    # Text analysis
                    text_result = fake_detector.analyze_credibility(news_text, news_url)
                    
                    # Image analysis
                    image_result = None
                    image_path = None
                    if uploaded_image:
                        image = Image.open(uploaded_image)
                        image_result = image_detector.analyze_image(image)
                        image_path = uploaded_image.name
                    
                    # Combined verdict
                    if image_result:
                        combined_fake_prob = (text_result['fake_probability'] + image_result['ai_probability']) / 2
                    else:
                        combined_fake_prob = text_result['fake_probability']
                    
                    if combined_fake_prob > 0.7:
                        final_verdict = "Likely Fake"
                    elif combined_fake_prob > 0.4:
                        final_verdict = "Suspicious"
                    else:
                        final_verdict = "Likely Real"
                    
                    # Save to database
                    db.save_fake_detection(
                        st.session_state.user_id,
                        news_text,
                        news_url,
                        image_path,
                        text_result['fake_probability'],
                        image_result['ai_probability'] if image_result else None,
                        final_verdict
                    )
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 🎯 Analysis Results")
                    
                    # Overall verdict
                    verdict_class = "verdict-real" if final_verdict == "Likely Real" else "verdict-fake" if final_verdict == "Likely Fake" else "verdict-suspicious"
                    st.markdown(f'<div class="{verdict_class}">Overall Verdict: {final_verdict}</div>', unsafe_allow_html=True)
                    
                    # Metrics
                    st.markdown("### 📊 Credibility Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Credibility Score",
                            f"{text_result['credibility_score']:.1f}/100",
                            delta=f"{text_result['credibility_score'] - 50:.1f}" if text_result['credibility_score'] != 50 else "0"
                        )
                    
                    with col2:
                        st.metric(
                            "Fake Probability",
                            f"{text_result['fake_probability']*100:.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Real Probability",
                            f"{text_result['real_probability']*100:.1f}%"
                        )
                    
                    # Text analysis details
                    st.markdown("### 📝 Text Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Probability gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=text_result['fake_probability'] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Fake News Probability"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgreen"},
                                    {'range': [40, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Text Features:**")
                        features = text_result['text_features']
                        st.write(f"- Word Count: {features['word_count']}")
                        st.write(f"- Text Length: {features['text_length']}")
                        st.write(f"- Exclamation Marks: {features['exclamation_count']}")
                        st.write(f"- Question Marks: {features['question_count']}")
                        st.write(f"- Caps Ratio: {features['caps_ratio']*100:.1f}%")
                        st.write(f"- Sensational Words: {features['sensational_count']}")
                    
                    # Warnings
                    if text_result['warnings']:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("**⚠️ Warning Signs Detected:**")
                        for warning in text_result['warnings']:
                            st.markdown(f"- {warning}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Image analysis
                    if image_result:
                        st.markdown("### 📸 Image Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # AI probability gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=image_result['ai_probability'] * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "AI-Generated Probability"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 40], 'color': "lightgreen"},
                                        {'range': [40, 70], 'color': "yellow"},
                                        {'range': [70, 100], 'color': "lightcoral"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"**Image Verdict:** {image_result['verdict']}")
                            st.markdown(f"**Confidence:** {image_result['confidence']}")
                            
                            if image_result['analysis_notes']:
                                st.markdown("**Analysis Notes:**")
                                for note in image_result['analysis_notes']:
                                    st.markdown(f"- {note}")
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    if final_verdict == "Likely Fake":
                        st.error("""
                        **This content shows strong signs of being fake news:**
                        - Cross-reference with reputable news sources
                        - Check the author and publication credibility
                        - Look for original sources and citations
                        - Be skeptical of sensational claims
                        - Don't share without verification
                        """)
                    elif final_verdict == "Suspicious":
                        st.warning("""
                        **This content requires further verification:**
                        - Verify claims with multiple sources
                        - Check for bias or sensationalism
                        - Look for supporting evidence
                        - Consider the context and timing
                        - Exercise caution before sharing
                        """)
                    else:
                        st.success("""
                        **This content appears credible, but always:**
                        - Verify important claims independently
                        - Consider multiple perspectives
                        - Check the publication date
                        - Look for primary sources
                        - Stay informed and critical
                        """)
            else:
                st.error("Please enter news text with at least 20 characters")
    
    with tab2:
        st.markdown("### 📊 Detection Statistics")
        
        stats = db.get_user_stats(st.session_state.user_id)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Detections", stats['total_detections'])
        
        with col2:
            if stats['total_detections'] > 0:
                recent = db.get_user_detections(st.session_state.user_id, limit=100)
                fake_count = sum(1 for r in recent if r['verdict'] == "Likely Fake")
                st.metric("Fake News Detected", fake_count)
            else:
                st.metric("Fake News Detected", 0)
        
        with col3:
            if stats['total_detections'] > 0:
                suspicious_count = sum(1 for r in recent if r['verdict'] == "Suspicious")
                st.metric("Suspicious Content", suspicious_count)
            else:
                st.metric("Suspicious Content", 0)
        
        if stats['total_detections'] > 0:
            # Verdict distribution
            recent = db.get_user_detections(st.session_state.user_id, limit=100)
            verdict_counts = {}
            for r in recent:
                verdict = r['verdict']
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(verdict_counts.keys()),
                    values=list(verdict_counts.values()),
                    marker_colors=['#2ecc71', '#f39c12', '#e74c3c']
                )
            ])
            fig.update_layout(title="Verdict Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📜 Detection History")
        
        history = db.get_user_detections(st.session_state.user_id, limit=20)
        
        if history:
            for i, record in enumerate(history):
                verdict_emoji = "✅" if record['verdict'] == "Likely Real" else "❌" if record['verdict'] == "Likely Fake" else "⚠️"
                
                with st.expander(f"{verdict_emoji} Detection #{len(history)-i} - {record['verdict']} ({record['created_at']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown("**Text Preview:**")
                        st.text(record['news_text'][:200] + "..." if len(record['news_text']) > 200 else record['news_text'])
                        if record['news_url']:
                            st.markdown(f"**URL:** {record['news_url']}")
                    
                    with col2:
                        st.markdown(f"**Verdict:** {record['verdict']}")
                        st.markdown(f"**Fake Probability:** {record['fake_probability']*100:.1f}%")
                        if record['ai_image_probability']:
                            st.markdown(f"**AI Image Prob:** {record['ai_image_probability']*100:.1f}%")
                        st.markdown(f"**Date:** {record['created_at']}")
        else:
            st.info("No detection history yet. Start analyzing news to build your history!")

if __name__ == "__main__":
    main()