from PIL import Image
import numpy as np
import config
import streamlit as st
from datetime import datetime
import random

class NewsGenerator:
    def __init__(self):
        """
        In production, use:
        - BLIP for image captioning
        - GPT-2/T5 for text generation
        
        For now, using heuristic approach
        """
        self.caption_templates = [
            "A photo showing {}",
            "An image depicting {}",
            "A scene featuring {}",
            "A view of {}"
        ]
        
        self.news_templates = {
            "people": [
                "Local residents gathered today for {}. The event, which took place {}, attracted significant attention from the community.",
                "In a developing story, {} was witnessed by several people {}. Authorities are monitoring the situation closely.",
                "Community members participated in {} earlier today. The gathering, described as peaceful, highlighted important local concerns."
            ],
            "building": [
                "A notable architectural feature {} has become the center of attention. Local officials confirmed that {}.",
                "The structure located {} represents a significant development in the area. Experts suggest that this could impact {}.",
                "This landmark {} has been making headlines. According to sources, the significance lies in {}."
            ],
            "nature": [
                "Natural phenomena observed {} have caught the attention of researchers. Scientists indicate that {}.",
                "The scenic view {} showcases the region's natural beauty. Environmental experts note that {}.",
                "Weather conditions {} created a remarkable sight. Meteorologists explain that this is {}."
            ],
            "vehicle": [
                "Transportation developments {} are making news today. Officials report that {}.",
                "A vehicle incident {} has prompted discussions about safety. Local authorities state that {}.",
                "Traffic patterns {} have been affected by recent changes. Transportation department confirms that {}."
            ],
            "object": [
                "An interesting discovery {} has emerged in recent reports. Experts are examining {}.",
                "The item found {} has raised questions among locals. Preliminary investigation suggests {}.",
                "Unusual circumstances {} were documented earlier. Officials are working to understand {}."
            ],
            "animal": [
                "Wildlife sighting {} has excited nature enthusiasts. Conservation experts note that {}.",
                "Animal behavior observed {} provides insights into local ecosystem. Researchers indicate that {}.",
                "The creature spotted {} represents an important ecological indicator. Scientists suggest that {}."
            ]
        }
    
    def analyze_image(self, image):
        """Analyze image to extract basic information"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Get image dimensions
        height, width = img_array.shape[:2]
        
        # Analyze colors
        avg_color = np.mean(img_array, axis=(0, 1))
        brightness = np.mean(avg_color)
        
        # Simple heuristics for content detection
        # In production, use BLIP or similar model
        
        # Determine dominant colors
        if avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
            color_desc = "blue tones"
        elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
            color_desc = "green tones"
        elif avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
            color_desc = "red tones"
        else:
            color_desc = "neutral tones"
        
        # Determine likely category based on brightness and colors
        if brightness > 150:
            category = "nature"
        elif brightness < 80:
            category = "building"
        else:
            category = random.choice(["people", "object", "vehicle"])
        
        return {
            'category': category,
            'brightness': brightness,
            'color_description': color_desc,
            'dimensions': (width, height)
        }
    
    def generate_caption(self, image_info):
        """Generate caption from image analysis"""
        category = image_info['category']
        color_desc = image_info['color_description']
        
        # Simple caption generation
        objects = {
            "people": ["a group of people", "individuals", "a crowd", "community members"],
            "building": ["a building", "an architectural structure", "a facility", "infrastructure"],
            "nature": ["natural scenery", "outdoor environment", "landscape", "natural features"],
            "vehicle": ["a vehicle", "transportation", "a mode of transport", "machinery"],
            "object": ["various objects", "items of interest", "distinctive features", "notable elements"],
            "animal": ["wildlife", "an animal", "fauna", "a creature"]
        }
        
        obj = random.choice(objects[category])
        template = random.choice(self.caption_templates)
        caption = template.format(f"{obj} with {color_desc}")
        
        return caption
    
    def generate_title(self, caption, category):
        """Generate news title"""
        title_templates = {
            "people": [
                "Local Community Event Draws Attention",
                "Residents Gather for Important Discussion",
                "Community Activity Highlights Local Issues",
                "Public Gathering Addresses Key Concerns"
            ],
            "building": [
                "Architectural Development Makes Headlines",
                "New Structure Becomes Local Landmark",
                "Building Project Progresses in Region",
                "Infrastructure Update Announced"
            ],
            "nature": [
                "Natural Phenomenon Observed in Area",
                "Environmental Conditions Create Spectacular View",
                "Nature's Display Captivates Observers",
                "Scenic Beauty Highlights Regional Features"
            ],
            "vehicle": [
                "Transportation Update Affects Local Routes",
                "Vehicle Development Announced",
                "Traffic Changes Implemented",
                "Transportation News Emerges"
            ],
            "object": [
                "Interesting Discovery Made in Region",
                "Unusual Find Raises Questions",
                "Local Discovery Sparks Interest",
                "Notable Item Identified"
            ],
            "animal": [
                "Wildlife Sighting Excites Locals",
                "Animal Behavior Provides Insights",
                "Nature Encounter Documented",
                "Wildlife Activity Observed"
            ]
        }
        
        return random.choice(title_templates[category])
    
    def generate_content(self, caption, category):
        """Generate full news article content"""
        templates = self.news_templates[category]
        
        # Generate multiple paragraphs
        paragraphs = []
        
        # Opening paragraph
        location = random.choice(["in the downtown area", "near the city center", 
                                 "in a residential neighborhood", "at a local venue"])
        time = random.choice(["this morning", "earlier today", "this afternoon", "recently"])
        
        opening = random.choice(templates).format(location, time)
        paragraphs.append(opening)
        
        # Middle paragraph with details
        details = [
            "Witnesses reported seeing the event unfold with great interest. Local authorities have been informed and are assessing the situation.",
            "The incident has sparked conversations among community members. Officials stated that they are monitoring developments closely.",
            "Observers noted the significance of this occurrence. Experts suggest this could have broader implications for the area.",
            "The situation developed over several hours. Sources indicate that response teams were quick to arrive on scene."
        ]
        paragraphs.append(random.choice(details))
        
        # Closing paragraph
        conclusions = [
            "Further updates are expected as more information becomes available. Residents are advised to stay informed through official channels.",
            "The community continues to observe the situation with interest. Local officials promise to provide updates as they emerge.",
            "As the story develops, authorities encourage public awareness. More details will be released in the coming days.",
            "Investigation into the matter is ongoing. The public will be notified of any significant developments."
        ]
        paragraphs.append(random.choice(conclusions))
        
        return "\n\n".join(paragraphs)
    
    def generate_news(self, image):
        """Generate complete news article from image"""
        # Analyze image
        image_info = self.analyze_image(image)
        
        # Generate caption
        caption = self.generate_caption(image_info)
        
        # Generate title
        title = self.generate_title(caption, image_info['category'])
        
        # Generate content
        content = self.generate_content(caption, image_info['category'])
        
        # Add metadata
        return {
            'title': title,
            'caption': caption,
            'content': content,
            'category': image_info['category'],
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_info': image_info
        }