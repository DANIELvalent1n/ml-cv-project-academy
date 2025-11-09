import torch
import google.generativeai as genai
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NewsGenerator:
    def __init__(self):
        self.device = DEVICE
        self.blip_caption_proc = None
        self.blip_caption_model = None
        self.blip_vqa_proc = None
        self.blip_vqa_model = None
        self.gemini_model = None
    
    def load_blip_caption(self):
        if self.blip_caption_model is None:
            self.blip_caption_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.blip_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            self.blip_caption_model.to(self.device)
            self.blip_caption_model.eval()
        return self.blip_caption_proc, self.blip_caption_model
    
    def load_blip_vqa(self):
        if self.blip_vqa_model is None:
            self.blip_vqa_proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.blip_vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            self.blip_vqa_model.to(self.device)
            self.blip_vqa_model.eval()
        return self.blip_vqa_proc, self.blip_vqa_model
    
    def load_gemini(self, api_key):
        if self.gemini_model is None:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        return self.gemini_model
    
    def generate_caption(self, pil_img, max_length=64):
        proc, model = self.load_blip_caption()
        inputs = proc(pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=max_length)
        caption = proc.decode(out[0], skip_special_tokens=True)
        return caption
    
    def answer_vqa(self, pil_img, question, max_length=32):
        proc, model = self.load_blip_vqa()
        inputs = proc(images=pil_img, text=question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=max_length)
        answer = proc.decode(out[0], skip_special_tokens=True)
        return answer
    
    def generate_article(self, caption, vqa_dict, api_key):
        subject = vqa_dict.get("What is the main subject or focus of the image?", "Incident")
        objects = vqa_dict.get("What specific objects or items are prominently shown?", "")
        condition = vqa_dict.get("What condition or state are the main objects in?", "visible")
        people = vqa_dict.get("How many people are visible and what are they doing?", "")
        event = vqa_dict.get("What appears to have happened or is currently happening?", "occurred")
        location = vqa_dict.get("Is this indoors or outdoors?", "location")
        colors = vqa_dict.get("What colors and lighting dominate the scene?", "")
        time = vqa_dict.get("What time period or season does this suggest?", "")
        
        model = self.load_gemini(api_key)
        
        prompt_text = (
            f"Write a medium-long news article with these details:\n"
            f"Description: {caption}\n"
            f"Subject: {subject}\n"
            f"Objects: {objects}\n"
            f"Condition: {condition}\n"
            f"People: {people}\n"
            f"Event: {event}\n"
            f"Location: {location}\n"
            f"Colors: {colors}\n"
            f"Time: {time}\n\n"
            f"Tone: factual, neutral. Use 'a witness' or 'official' not real names. Format with TITLE, LEAD, BODY sections."
        )
        
        response = model.generate_content(
            prompt_text,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 300
            }
        )
        
        return response.text
    
    @staticmethod
    def parse_vqa_from_prompt(prompt_text):
        vqa_dict = {}
        lines = prompt_text.split('\n')
        
        for line in lines:
            if line.startswith("- "):
                parts = line[2:].split(": ", 1)
                if len(parts) == 2:
                    vqa_dict[parts[0]] = parts[1]
        
        return vqa_dict