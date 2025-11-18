import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import os
import json
from datetime import datetime

class ImageGenerator:
    """
    Handles model loading (with GPU/CPU fallback), image generation, 
    watermarking (Ethical AI), and metadata saving/exporting (Storage requirement).
    """
    def __init__(self):
        # Hardware Check: GPU preferred, CPU fallback (Requirement)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...")

        # Load Open-Source Model (Stable Diffusion v1.5)
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # --- ROBUST MODEL LOADING: Load model without specific dtype initially ---
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id)
        
        if self.device == "cuda":
            # Set data type for GPU operation (float16 for speed)
            self.pipe.to(torch.float16)
            
            # --- CRITICAL FIX: Graceful failure for xformers (Prevents crash) ---
            try:
                # Try to import xformers. If it fails, the app still runs.
                import xformers 
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xformers enabled for performance.")
            except ImportError:
                print("xformers not installed. Running with standard memory usage on GPU.")
            # --- END OF FIX ---
            
        elif self.device == "cpu":
            # CPU uses float32
            self.pipe.to(torch.float32)
            
        self.pipe.to(self.device)
        # Note on Ethical AI: The Stable Diffusion pipeline includes a safety checker
        # by default to filter inappropriate content, fulfilling the content filtering requirement.

    def generate(self, prompt, negative_prompt, steps=50, guidance=7.5, num_images=1, callback=None):
        """
        Generates images and supports a callback for real-time progress updates.
        """
        
        # Define a callback function compatible with Diffusers to track progress (Requirement)
        def progress_callback(step, timestep, latents):
            if callback:
                # Calculate percentage completed
                progress = step / steps
                callback(progress)

        # Generate images based on adjustable parameters 
        images = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_images, # Adjustable parameter (Requirement)
            callback=progress_callback,       # Progress update (Requirement)
            callback_steps=1
        ).images
        
        return images

    def add_watermark(self, image: Image.Image) -> Image.Image:
        """
        Adds a small watermark to the image, fulfilling the Ethical AI watermarking requirement.
        """
        draw = ImageDraw.Draw(image)
        text = "AI Generated | Talrn"
        
        width, height = image.size
        # Simple font fallback
        try:
            font = ImageFont.truetype("arial.ttf", 20) 
        except IOError:
            font = ImageFont.load_default()

        # Measure text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Padding from the edges
        margin = 10
        position = (width - text_width - margin, height - text_height - margin)
        
        # Draw the text in white
        draw.text(position, text, fill=(255, 255, 255), font=font) 
        return image

    def save_image(self, image: Image.Image, prompt: str, params: dict, index: int = 0) -> str:
        """
        Saves the image in PNG and JPEG formats, and saves a corresponding 
        JSON file with metadata (Storage and Export Requirement).
        
        Returns the basename for file retrieval.
        """
        output_dir = "generated_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use index to save multiple images uniquely
        filename_base = f"{output_dir}/img_{timestamp}_{index}"
        
        # Save Image (PNG) (Requirement: Multiple formats)
        filepath_png = f"{filename_base}.png"
        image.save(filepath_png, "PNG")
        
        # Save Image (JPEG) (Requirement: Multiple formats)
        # Convert to RGB before saving as JPEG as some SD outputs are RGBA
        filepath_jpeg = f"{filename_base}.jpeg"
        image.convert("RGB").save(filepath_jpeg, "JPEG", quality=95)
        
        # Save Metadata (JSON)
        metadata = {
            "prompt": prompt,
            "timestamp": timestamp,
            "parameters": params,
            "model": "Stable Diffusion v1.5",
            "device": self.device
        }
        filepath_json = f"{filename_base}.json"
        with open(filepath_json, "w") as f:
            json.dump(metadata, f, indent=4)
            
        return filename_base # Return base name for app.py to construct paths