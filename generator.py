import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import os
import json
from datetime import datetime

class ImageGenerator:
    def __init__(self):
        # Hardware Check: GPU preferred, CPU fallback [cite: 10, 31]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...")

        # Load Open-Source Model (Stable Diffusion v1.5) [cite: 9]
        model_id = "runwayml/stable-diffusion-v1-5"
        
        if self.device == "cuda":
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        else:
            # CPU does not support float16 well, use float32
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

        self.pipe = self.pipe.to(self.device)
        
        # Ensure Safety Checker is ON for Ethical AI [cite: 32, 33]
        # self.pipe.safety_checker = ... (Enabled by default in diffusers)

    def generate(self, prompt, negative_prompt, steps=50, guidance=7.5):
        # Generate image based on adjustable parameters [cite: 13]
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]
        return image

    def add_watermark(self, image):
        # Ethical AI: Watermarking [cite: 33]
        draw = ImageDraw.Draw(image)
        text = "AI Generated"
        # You might need to load a specific font file, or use default
        # font = ImageFont.truetype("arial.ttf", 15) 
        draw.text((10, 10), text, fill=(255, 255, 255)) 
        return image

    def save_image(self, image, prompt, params):
        # Storage: Save metadata and image in organized folders [cite: 26]
        if not os.path.exists("generated_images"):
            os.makedirs("generated_images")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"generated_images/img_{timestamp}"
        
        # Save Image
        image.save(f"{filename_base}.png")
        
        # Save Metadata [cite: 26]
        metadata = {
            "prompt": prompt,
            "timestamp": timestamp,
            "parameters": params
        }
        with open(f"{filename_base}.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        return f"{filename_base}.png"