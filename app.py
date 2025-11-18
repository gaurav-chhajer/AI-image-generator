import streamlit as st
from generator import ImageGenerator

# Initialize the model once (cache it so it doesn't reload every click)
@st.cache_resource
def get_generator():
    return ImageGenerator()

generator = get_generator()

st.title("AI-Powered Image Generator")
st.markdown("### Internship Task Implementation [cite: 4]")

# Sidebar: Configuration [cite: 13, 19]
st.sidebar.header("Generation Settings")
steps = st.sidebar.slider("Inference Steps (Quality)", 10, 100, 50)
guidance = st.sidebar.slider("Guidance Scale ( adherence to prompt)", 1.0, 20.0, 7.5)
style = st.sidebar.selectbox("Style Preset", ["None", "Cinematic", "Anime", "Oil Painting"])

# Main Input [cite: 18]
prompt = st.text_area("Enter your prompt:", "A futuristic city at sunset")
negative_prompt = st.text_input("Negative Prompt (What to avoid):", "blurry, low quality, distorted")

# Prompt Engineering Logic [cite: 23]
if style == "Cinematic":
    final_prompt = f"{prompt}, 4k, highly detailed, dramatic lighting, shallow depth of field"
elif style == "Anime":
    final_prompt = f"{prompt}, anime style, studio ghibli, vibrant colors"
elif style == "Oil Painting":
    final_prompt = f"{prompt}, oil painting, thick brushstrokes, van gogh style"
else:
    final_prompt = prompt

# Progress and Generation [cite: 21]
if st.button("Generate Image"):
    with st.spinner("Generating... (Check terminal for progress)"):
        try:
            # 1. Generate
            image = generator.generate(final_prompt, negative_prompt, steps, guidance)
            
            # 2. Apply Watermark [cite: 33]
            image = generator.add_watermark(image)
            
            # 3. Display Image [cite: 20]
            st.image(image, caption="Generated Output", use_column_width=True)
            
            # 4. Save & Metadata [cite: 26]
            filepath = generator.save_image(image, final_prompt, {"steps": steps, "style": style})
            st.success(f"Image saved to {filepath}")
            
            # 5. Download Button [cite: 20]
            with open(filepath, "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")