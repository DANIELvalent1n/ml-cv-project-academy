import streamlit as st
from PIL import Image
import torch
import traceback

try:
    import hf_xet
except Exception:
    pass

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

st.set_page_config(page_title="Image → Article", layout="wide")
st.title("📰 Image → Article")

if "caption" not in st.session_state:
    st.session_state.caption = None
if "vqa_answers" not in st.session_state:
    st.session_state.vqa_answers = {}
if "prompt_en" not in st.session_state:
    st.session_state.prompt_en = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Device: **{DEVICE}**")

@st.cache_resource(show_spinner=False)
def load_blip_caption():
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.to(DEVICE)
    model.eval()
    return proc, model

@st.cache_resource(show_spinner=False)
def load_blip_vqa():
    proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model.to(DEVICE)
    model.eval()
    return proc, model

@st.cache_resource(show_spinner=False)
def load_local_model():
    from transformers import pipeline
    generator = pipeline("text-generation", model="gpt2-xl", device=-1)
    return generator

def run_caption(proc, model, pil_img, max_length=64):
    inputs = proc(pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_length)
    caption = proc.decode(out[0], skip_special_tokens=True)
    return caption

def run_vqa(proc, model, pil_img, question, max_length=32):
    inputs = proc(images=pil_img, text=question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_length)
    answer = proc.decode(out[0], skip_special_tokens=True)
    return answer

def assemble_prompt_en(caption, vqa_answers: dict):
    lines = []
    lines.append(f"Brief image description: {caption}.")
    lines.append("Detailed answers to image questions:")
    for q, a in vqa_answers.items():
        lines.append(f"- {q}: {a}")
    prompt = "\n".join(lines)
    return prompt

def generate_article(generator, prompt):
    caption = ""
    vqa_dict = {}
    
    lines = prompt.split('\n')
    for line in lines:
        if "Brief image description:" in line:
            caption = line.split("Brief image description:")[1].strip().rstrip(".")
        elif line.startswith("- "):
            parts = line[2:].split(": ", 1)
            if len(parts) == 2:
                vqa_dict[parts[0]] = parts[1]
    
    subject = vqa_dict.get("What is the main subject or focus of the image?", "Incident")
    objects = vqa_dict.get("What specific objects or items are prominently shown?", "")
    condition = vqa_dict.get("What condition or state are the main objects in?", "visible")
    people = vqa_dict.get("How many people are visible and what are they doing?", "")
    event = vqa_dict.get("What appears to have happened or is currently happening?", "occurred")
    location = vqa_dict.get("Is this indoors or outdoors?", "location")
    colors = vqa_dict.get("What colors and lighting dominate the scene?", "")
    time = vqa_dict.get("What time period or season does this suggest?", "")
    
    generation_prompt = (
        f"Write a news article:\n"
        f"TITLE: {subject}\n"
        f"LEAD: One sentence summary about the incident.\n"
        f"BODY: Subject: {subject}. Objects: {objects}. Condition: {condition}. "
        f"People: {people}. Event: {event}. Location: {location}. Colors: {colors}. Time: {time}.\n"
        f"Tone: factual, neutral. Use 'a witness' or 'official' not real names. ~150 words.\n\n"
    )
    
    outputs = generator(
        generation_prompt,
        max_length=500,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.2
    )
    
    text = outputs[0]["generated_text"].replace(generation_prompt, "").strip()
    
    if not text or len(text) < 15:
        raise Exception("Model failed to generate meaningful text.")
    
    article = f"""TITLE:
{subject}

LEAD:
{event.capitalize()} in {location}.

BODY:
{condition}. {event}. {text}

---
Authorities are investigating. Further updates to follow."""
    
    return article

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="uploader")

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded", use_column_width=True)

    if st.session_state.caption is None:
        if st.button("Generate Caption + VQA", key="btn_caption"):
            proc_caption, blip_caption_model = load_blip_caption()
            proc_vqa, blip_vqa_model = load_blip_vqa()

            with st.spinner("Generating caption..."):
                caption = run_caption(proc_caption, blip_caption_model, pil_img)
                st.session_state.caption = caption
                st.success("Caption generated!")

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

            vqa_answers = {}
            st.write("🔎 VQA Answers:")
            for q in questions:
                ans = run_vqa(proc_vqa, blip_vqa_model, pil_img, q)
                vqa_answers[q] = ans
                st.write(f"- **{q}** → {ans}")

            st.session_state.vqa_answers = vqa_answers
            prompt_en = assemble_prompt_en(st.session_state.caption, st.session_state.vqa_answers)
            st.session_state.prompt_en = prompt_en
            
            st.rerun()

    else:
        st.success(f"✅ Caption: {st.session_state.caption}")
        st.write(f"✅ VQA answers: {len(st.session_state.vqa_answers)} questions")
        
        if st.button("Generate Article", key="btn_generate"):
            generator = load_local_model()
            article = generate_article(generator, st.session_state.prompt_en)
            st.write(article)

            if st.button("Save Article", key="btn_save"):
                from app.utils import insert_article
                aid = insert_article(title="Auto-generated", content=article, image_path=None, user_id=1)
                st.success(f"Saved with id={aid}")
        
        if st.button("Analyze Another Image", key="btn_reset"):
            st.session_state.caption = None
            st.session_state.vqa_answers = {}
            st.session_state.prompt_en = None
            st.rerun()