import streamlit as st
from PIL import Image
import traceback
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.news_generator import NewsGenerator
from utils.auth import init_session_state, require_auth
from database.db_manager import DatabaseManager
import plotly.graph_objects as go

try:
    import hf_xet
except Exception:
    pass

st.set_page_config(
    page_title="📰 News Generator",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
<style>
    .article-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
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

init_session_state()

@require_auth
def main():
    st.title("📰 News Generator - Image to Article")
    st.markdown("Transform images into professional news articles using AI")
    
    if not st.session_state.logged_in:
        st.warning("❌ Please log in to use this feature")
        st.stop()
    
    st.success(f"✅ Welcome, {st.session_state.user['username']}")
    
    if "caption" not in st.session_state:
        st.session_state.caption = None
    if "vqa_answers" not in st.session_state:
        st.session_state.vqa_answers = {}
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    
    db = DatabaseManager()
    
    tab1, tab2, tab3 = st.tabs(["📸 Generate Article", "📊 Statistics", "📜 History"])
    
    with tab1:
        st.markdown("### Create News Articles from Images")
        
        col_upload, col_camera = st.columns([1, 1])
        
        with col_upload:
            st.markdown("**📁 Upload Images**")
            uploaded_images = st.file_uploader(
                "Upload images (up to 3)",
                type=["jpg", "jpeg", "png"],
                key="uploader",
                accept_multiple_files=True,
                help="Upload 1-3 images to analyze and generate news article"
            )
        
        with col_camera:
            st.markdown("**📷 Capture from Camera**")
            st.info(f"📸 Captured: {len(st.session_state.get('captured_images', []))} photo(s)")
            
            captured_image = st.camera_input(
                "Take a photo",
                help="Capture an image directly from your device camera"
            )
            
            if captured_image:
                if "captured_images" not in st.session_state:
                    st.session_state.captured_images = []
                
                if len(st.session_state.captured_images) < 3:
                    st.session_state.captured_images.append(captured_image)
                    st.success(f"✅ Photo added! ({len(st.session_state.captured_images)}/3)")
                else:
                    st.error("❌ Maximum 3 images reached")
            
            if st.session_state.get("captured_images"):
                if st.button("🗑️ Clear Captured Photos", use_container_width=True):
                    st.session_state.captured_images = []
                    st.rerun()
        
        # Combine uploaded and captured images
        all_images = list(uploaded_images) if uploaded_images else []
        if "captured_images" in st.session_state:
            all_images.extend(st.session_state.captured_images)
        
        if len(all_images) > 3:
            st.error("❌ Maximum 3 images allowed. Please remove some images.")
            all_images = all_images[:3]
        
        uploaded_images = all_images if all_images else None
        
        if uploaded_images:
            if len(uploaded_images) > 3:
                st.error("❌ Maximum 3 images allowed. Please upload up to 3 images.")
                uploaded_images = uploaded_images[:3]
            
            # Initialize session state for multiple images
            if "processed_images" not in st.session_state:
                st.session_state.processed_images = {}
            if "combined_vqa" not in st.session_state:
                st.session_state.combined_vqa = {}
            if "all_captions" not in st.session_state:
                st.session_state.all_captions = []
            
            # Display all uploaded images
            cols = st.columns(min(len(uploaded_images), 3))
            for idx, uploaded in enumerate(uploaded_images):
                with cols[idx]:
                    pil_img = Image.open(uploaded).convert("RGB")
                    st.image(pil_img, caption=f"Image {idx+1}", use_container_width=True)
                    st.markdown(f"**{uploaded.name}**")
                    st.markdown(f"- Size: {pil_img.size[0]}x{pil_img.size[1]}px")
            
            st.markdown("---")
            
            if st.session_state.caption is None:
                if st.button(f"🔍 Analyze {len(uploaded_images)} Image(s) (Caption + VQA)", key="btn_caption", use_container_width=True, type="primary"):
                    generator = NewsGenerator()
                    
                    questions = [
                        "What is the main subject or focus of the image?",
                        "What specific objects or items are prominently shown?",
                        "What condition or state are the main objects in?",
                        "How many people are visible and what are they doing?",
                        "What appears to have happened or is currently happening?",
                        "Is this indoors or outdoors?",
                        "What colors and lighting dominate the scene?",
                        "What time period or season does this suggest?"
                    ]
                    
                    total_steps = len(uploaded_images) * (1 + len(questions))
                    current_step = 0
                    progress_bar = st.progress(0)
                    
                    combined_vqa = {}
                    all_captions = []
                    
                    for img_idx, uploaded in enumerate(uploaded_images):
                        st.markdown(f"**Processing Image {img_idx + 1}/{len(uploaded_images)}...**")
                        
                        pil_img = Image.open(uploaded).convert("RGB")
                        
                        # Generate caption
                        with st.spinner(f"📝 Generating caption for image {img_idx + 1}..."):
                            try:
                                caption = generator.generate_caption(pil_img)
                                all_captions.append(caption)
                                current_step += 1
                                progress_bar.progress(current_step / total_steps)
                            except Exception as e:
                                st.error(f"❌ Error generating caption for image {img_idx + 1}: {e}")
                                st.stop()
                        
                        # Generate VQA for each image
                        with st.spinner(f"🤖 Analyzing image {img_idx + 1} details..."):
                            try:
                                for q in questions:
                                    ans = generator.answer_vqa(pil_img, q)
                                    
                                    # Combine answers from multiple images
                                    key = f"Image {img_idx + 1} - {q}"
                                    combined_vqa[key] = ans
                                    
                                    current_step += 1
                                    progress_bar.progress(current_step / total_steps)
                            except Exception as e:
                                st.error(f"❌ Error during VQA for image {img_idx + 1}: {e}")
                                st.stop()
                    
                    st.session_state.caption = " | ".join(all_captions)
                    st.session_state.all_captions = all_captions
                    st.session_state.combined_vqa = combined_vqa
                    st.session_state.vqa_answers = combined_vqa
                    st.session_state.num_images = len(uploaded_images)
                    
                    st.success(f"✅ Analysis of {len(uploaded_images)} image(s) complete!")
                    st.rerun()
            
            else:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown(f"**✅ {len(st.session_state.all_captions)} Caption(s) Generated**")
                    for idx, cap in enumerate(st.session_state.all_captions):
                        st.markdown(f"**Image {idx+1}:** {cap[:60]}...")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**✅ Analysis Complete**")
                    st.markdown(f"- Images Analyzed: {len(st.session_state.all_captions)}")
                    st.markdown(f"- Details Extracted: {len(st.session_state.combined_vqa)}")
                    st.markdown("- Ready for article generation")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with st.expander(f"📋 View Detailed Analysis ({len(st.session_state.all_captions)} images)"):
                    for img_idx in range(len(st.session_state.all_captions)):
                        st.markdown(f"### Image {img_idx + 1}")
                        st.markdown(f"**Caption:** {st.session_state.all_captions[img_idx]}")
                        
                        for q, a in st.session_state.combined_vqa.items():
                            if f"Image {img_idx + 1}" in q:
                                question_text = q.replace(f"Image {img_idx + 1} - ", "")
                                st.markdown(f"**{question_text}**")
                                st.markdown(f"*{a}*")
                        st.divider()
                
                st.markdown("---")
                
                if st.button("✍️ Generate Article with Gemini", key="btn_generate", use_container_width=True, type="primary"):
                    generator = NewsGenerator()
                    
                    with st.spinner("📰 Generating news article..."):
                        try:
                            article = generator.generate_article(
                                st.session_state.caption,
                                st.session_state.vqa_answers,
                                st.session_state.gemini_api_key
                            )
                            
                            st.session_state.article = article
                            st.session_state.article_image = uploaded.name
                            
                        except Exception as e:
                            st.error(f"❌ Error generating article: {e}")
                            st.text(traceback.format_exc())
                            st.stop()
                    
                    st.success("✅ Article generated successfully!")
                
                if "article" in st.session_state:
                    st.markdown("---")
                    st.markdown('<div class="article-box">', unsafe_allow_html=True)
                    st.markdown("## 📄 Generated Article")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown(st.session_state.article)
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button("💾 Save Article", key="btn_save", use_container_width=True):
                            try:
                                db.save_generated_news(
                                    st.session_state.user_id,
                                    f"{len(st.session_state.all_captions)} images",
                                    " | ".join(st.session_state.all_captions),
                                    "Auto-generated News",
                                    st.session_state.article
                                )
                                st.success("✅ Article saved to your library!")
                            except Exception as e:
                                st.error(f"❌ Error saving: {e}")
               
                    with col2:
                        if st.button("🔄 Generate Another", key="btn_reset", use_container_width=True):
                            st.session_state.caption = None
                            st.session_state.vqa_answers = {}
                            if "article" in st.session_state:
                                del st.session_state.article
                            if "article_image" in st.session_state:
                                del st.session_state.article_image
                            st.rerun()
    
    with tab2:
        st.markdown("### 📊 Generation Statistics")
        
        stats = db.get_user_stats(st.session_state.user_id)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📰 Total Articles Generated",
                stats['total_generated'],
                delta=None
            )
        
        with col2:
            st.metric(
                "🔍 News Classifications",
                stats['total_classifications'],
                delta=None
            )
        
        with col3:
            st.metric(
                "🚨 Fake News Detections",
                stats['total_detections'],
                delta=None
            )
        
        if stats['total_generated'] > 0:
            st.markdown("---")
            st.markdown("### 📈 Generation Timeline")
            
            recent = db.get_user_generated_news(st.session_state.user_id, limit=30)
            
            if recent:
                dates = [r['created_at'][:10] for r in recent]
                daily_counts = {}
                for date in dates:
                    daily_counts[date] = daily_counts.get(date, 0) + 1
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(daily_counts.keys()),
                    y=list(daily_counts.values()),
                    mode='lines+markers',
                    name='Articles Generated',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title="Articles Generated Over Time",
                    xaxis_title="Date",
                    yaxis_title="Count",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📜 Generation History")
        
        history = db.get_user_generated_news(st.session_state.user_id, limit=20)
        
        if history:
            for i, record in enumerate(history):
                with st.expander(f"📰 Article #{len(history)-i} - {record['created_at'][:10]}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Caption:**")
                        st.text(record['image_caption'][:100] + "..." if len(record['image_caption'] or "") > 100 else record['image_caption'])
                        
                        st.markdown("**Article Preview:**")
                        st.text(record['generated_content'][:300] + "..." if len(record['generated_content']) > 300 else record['generated_content'])
                    
                    with col2:
                        st.markdown(f"**ID:** {record['id']}")
                        st.markdown(f"**Date:** {record['created_at']}")
                        
                        if st.button(f"👁️ View Full", key=f"view_{record['id']}", use_container_width=True):
                            st.markdown("---")
                            st.markdown(record['generated_content'])
        else:
            st.info("📭 No generation history yet. Start creating articles to build your history!")

if __name__ == "__main__":
    main()