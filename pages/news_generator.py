import streamlit as st
from PIL import Image
import traceback
from models.news_generator import NewsGenerator
from utils.auth import init_session_state, require_auth

try:
    import hf_xet
except Exception:
    pass

init_session_state()

@require_auth
def main():

    st.set_page_config(page_title="📰 News Generator", layout="wide")


    if not st.session_state.logged_in:
        st.warning("❌ Please log in to use this feature")
        st.stop()

    st.title("📰 News Generator - Image to Article")
    st.success(f"✅ Welcome, {st.session_state.user['username']}")

    if "caption" not in st.session_state:
        st.session_state.caption = None
    if "vqa_answers" not in st.session_state:
        st.session_state.vqa_answers = {}
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = st.secrets.get("GEMINI_API_KEY")

    st.write(f"Device: **GPU** ✅" if True else "Device: **CPU**")

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="uploader")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.info("Image loaded successfully")

        if st.session_state.caption is None:
            if st.button("🔍 Analyze Image (Caption + VQA)", key="btn_caption", use_container_width=True):
                generator = NewsGenerator()
                
                with st.spinner("📝 Generating caption..."):
                    try:
                        caption = generator.generate_caption(pil_img)
                        st.session_state.caption = caption
                    except Exception as e:
                        st.error(f"Error generating caption: {e}")
                        st.stop()
                
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
                
                with st.spinner("🤖 Analyzing image details..."):
                    vqa_answers = {}
                    try:
                        for q in questions:
                            ans = generator.answer_vqa(pil_img, q)
                            vqa_answers[q] = ans
                    except Exception as e:
                        st.error(f"Error during VQA: {e}")
                        st.stop()
                
                st.session_state.vqa_answers = vqa_answers
                st.rerun()
        
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.success(f"✅ Caption: {st.session_state.caption}")
            
            with col2:
                st.info(f"✅ {len(st.session_state.vqa_answers)} details analyzed")
            
            with st.expander("📋 View Analysis Details"):
                for q, a in st.session_state.vqa_answers.items():
                    st.write(f"**{q}**\n{a}")
            
            if st.button("✍️ Generate Article", key="btn_generate", use_container_width=True):
                generator = NewsGenerator()
                
                with st.spinner("📰 Generating news article with Gemini..."):
                    try:
                        article = generator.generate_article(
                            st.session_state.caption,
                            st.session_state.vqa_answers,
                            st.session_state.gemini_api_key
                        )
                        
                        st.markdown("---")
                        st.markdown(article)
                        st.markdown("---")
                        
                        st.session_state.article = article
                        
                    except Exception as e:
                        st.error(f"❌ Error generating article: {e}")
                        st.text(traceback.format_exc())
            
            if "article" in st.session_state:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("💾 Save Article", key="btn_save", use_container_width=True):
                        try:
                            from app.utils import insert_article
                            aid = insert_article(
                                title="Auto-generated News",
                                content=st.session_state.article,
                                image_path=None,
                                user_id=user['id']
                            )
                            st.success(f"✅ Article saved (ID: {aid})")
                        except Exception as e:
                            st.error(f"Error saving: {e}")
                
                with col2:
                    if st.button("🔄 Analyze Another Image", key="btn_reset", use_container_width=True):
                        st.session_state.caption = None
                        st.session_state.vqa_answers = {}
                        if "article" in st.session_state:
                            del st.session_state.article
                        st.rerun()

if __name__ == "__main__":
    main()