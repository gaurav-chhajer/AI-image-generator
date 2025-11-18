import streamlit as st
from generator import ImageGenerator
import os
import sys

# --- Configuration and Initialization ---

# Use an empty container to display initialization status, which is not cached
init_status = st.empty()

# Initialize the model once and cache it. THIS FUNCTION MUST NOT CALL ANY STREAMLIT UI ELEMENTS.
@st.cache_resource
def get_generator():
    """Initializes and returns the ImageGenerator instance."""
    # Use standard Python printing for status during initialization, which is safe inside @st.cache_resource
    print("Initializing Stable Diffusion Model...")
    try:
        gen = ImageGenerator()
        print(f"Model loaded successfully on {gen.device}!")
        return gen
    except Exception as e:
        # In case of critical failure, print the error and exit cleanly
        print(f"FATAL ERROR loading model: {e}", file=sys.stderr)
        return None

# Get the initialized generator object
generator = get_generator()

# Check if generator failed to load and stop execution
if generator is None:
    # The error message is already shown by the previous st.error call inside the failed try block.
    # We exit here to prevent subsequent code from crashing when accessing generator properties.
    st.stop()
else:
    # Display success message outside the cached function
    init_status.success(f"Model loaded successfully on {generator.device.upper()}!")


# --- Streamlit UI Layout ---

st.set_page_config(
    page_title="AI-Powered Image Generator (Talrn Internship Task)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖼️ AI-Powered Image Generator")
st.markdown("A text-to-image system built with Stable Diffusion and Streamlit to complete the internship assessment.")

# --- Sidebar: Configuration (Adjustable Parameters) ---

st.sidebar.header("Generation Settings")

# Adjustable parameter: Number of images per prompt (Requirement)
num_images = st.sidebar.slider("Number of Images", 1, 4, 1)

# Adjustable parameter: Inference Steps (impacts quality/time)
steps = st.sidebar.slider("Inference Steps (Quality/Time)", 10, 100, 50)

# Adjustable parameter: Guidance Scale (adherence to prompt)
guidance = st.sidebar.slider("Guidance Scale (Prompt Adherence)", 1.0, 20.0, 7.5)

# Adjustable parameter: Style guidance (Prompt Engineering Requirement)
style = st.sidebar.selectbox("Style Preset (Prompt Engineering)", 
    ["None", "Photorealistic", "Artistic (Van Gogh)", "Cartoon/Anime", "Cyberpunk"])

st.sidebar.markdown("---")
st.sidebar.info(f"Model running on: **{generator.device.upper()}**")


# --- Main Input Area ---

prompt = st.text_area("✍️ Enter your text description (Prompt):", "A futuristic city at sunset, highly detailed")
negative_prompt = st.text_input("🚫 Negative Prompt (What to avoid):", "blurry, low quality, distorted, watermark, signature")
st.caption("Use negative prompts to filter unwanted elements (Requirement).")


# --- Prompt Engineering Logic (Requirement) ---

final_prompt = prompt
if style == "Photorealistic":
    # Enhance prompt with descriptors for high quality
    final_prompt = f"{prompt}, highly detailed, 4K, professional photography, realistic lighting, octane render"
elif style == "Artistic (Van Gogh)":
    final_prompt = f"{prompt}, oil painting style, van gogh, thick brushstrokes, impressionism"
elif style == "Cartoon/Anime":
    final_prompt = f"{prompt}, studio ghibli style, cel-shaded, clean lines, vibrant colors, anime art"
elif style == "Cyberpunk":
    final_prompt = f"{prompt}, neon lights, dystopian, cinematic lighting, synthwave art, low-light"
    
if final_prompt != prompt:
    with st.expander("Final Prompt Used (After Engineering)"):
        st.code(final_prompt)

# --- Generation Trigger and Display (Handles Progress Bar and Multiple Downloads) ---

if st.button("🚀 Generate Images", use_container_width=True, type="primary"):
    
    # Placeholders for progress bar and status text (Requirement)
    progress_bar = st.progress(0, "Starting generation...")
    status_text = st.empty()
    
    # Define a function to update the UI from the backend (Requirement)
    def ui_callback(progress):
        """Updates the Streamlit progress bar and text."""
        percentage = int(progress * 100)
        progress_bar.progress(progress)
        status_text.text(f"Progress: {percentage}% | Step: {int(progress * steps)}/{steps}")

    try:
        # 1. Generate Images
        status_text.text("Generating... This may take a moment.")
        images = generator.generate(
            final_prompt, 
            negative_prompt, 
            steps, 
            guidance, 
            num_images,      # Passes the adjustable number of images
            callback=ui_callback # Passes the progress updater
        )
        
        # Clear progress bar when done
        progress_bar.empty()
        status_text.success("Generation Complete! Images and metadata saved locally.")
        
        # 2. Display and Save Loop (Handles Multiple Images)
        st.header("Generated Results")
        cols = st.columns(num_images) 
        
        # Pass the generation parameters for metadata logging
        params = {
            "steps": steps, 
            "guidance": guidance, 
            "style_preset": style
        }
        
        for i, img in enumerate(images):
            with cols[i]:
                st.subheader(f"Image {i+1}")
                
                # Apply Watermark (Ethical AI Requirement)
                watermarked_img = generator.add_watermark(img)
                
                # Display Image
                st.image(watermarked_img, use_column_width=True)
                
                # Saves PNG, JPEG, and JSON metadata (Storage and Export Requirement)
                filepath_base = generator.save_image(img, final_prompt, params, index=i)
                
                # Download Buttons (Multiple Formats Requirement)
                st.markdown("**Download:**")
                
                # PNG Download
                png_path = f"{filepath_base}.png"
                with open(png_path, "rb") as file:
                    st.download_button(
                        label="Download PNG",
                        data=file,
                        file_name=os.path.basename(png_path),
                        mime="image/png",
                        key=f"dl_png_{i}"
                    )
                
                # JPEG Download
                jpeg_path = f"{filepath_base}.jpeg"
                with open(jpeg_path, "rb") as file:
                    st.download_button(
                        label="Download JPEG",
                        data=file,
                        file_name=os.path.basename(jpeg_path),
                        mime="image/jpeg",
                        key=f"dl_jpeg_{i}"
                    )


    except Exception as e:
        # Display an error if anything goes wrong
        st.error(f"Generation failed. Check the console for model loading errors. Error: {e}")