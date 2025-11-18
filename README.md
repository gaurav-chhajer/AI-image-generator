# **AI-Powered Image Generator — Talrn ML Internship Task Assessment Submission**

This project is a complete, full-stack application for text-to-image generation, built to demonstrate proficiency in open-source generative AI models, Deep Learning (PyTorch), web development (Streamlit), and ethical AI design.

---

## **1. Project Overview and Architecture**

The system provides a user-friendly interface for generating images using a state-of-the-art diffusion model.
It follows a **three-layer architecture**:

### **Frontend (Streamlit – `app.py`)**

* Provides the interactive UI.
* Manages user input and persists all settings via `st.session_state`.
* Displays real-time progress indicators.

### **Backend (Python – `generator.py`)**

* Contains the `ImageGenerator` class.
* Handles model loading, device selection, image generation, watermarking, and file I/O.

### **Model (Stable Diffusion v1.5)**

* Core engine for text-to-image synthesis.

---

## **2. Technology Stack and Model Details**

| Component               | Technology                 | Role                                        |
| ----------------------- | -------------------------- | ------------------------------------------- |
| Deep Learning Framework | **PyTorch**                | Model execution & GPU acceleration          |
| Model                   | **Stable Diffusion v1.5**  | Latent diffusion text-to-image model        |
| Model Library           | **Hugging Face Diffusers** | High-level pipeline for loading & inference |
| Web Interface           | **Streamlit**              | UI development                              |
| Utility Library         | **PIL (Pillow)**           | Watermarking, image export                  |

---

## **3. Setup and Installation Steps**

Requirements: **Python 3.8+** and **pip**.

### **1. Clone the Repository**

```bash
git clone https://github.com/gaurav-chhajer/AI-image-generator.git
cd ai-image-generator
```

### **2. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux & macOS
# On Windows:
# venv\Scripts\activate
```

### **3. Install Dependencies**

> Your `requirements.txt` should include:
> `torch, diffusers, transformers, streamlit, accelerate, pillow`

---

### **If you have an NVIDIA GPU (Recommended):**

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers streamlit accelerate pillow
```

### **If you are using CPU or a non-NVIDIA GPU:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers streamlit accelerate pillow
```

---

### **4. Run the Application**

```bash
streamlit run app.py
```

---

## **4. Hardware Requirements (GPU/CPU Specifications)**

| Mode                | Minimum Requirement    | Performance             | Implementation (in `generator.py`) |
| ------------------- | ---------------------- | ----------------------- | ---------------------------------- |
| **GPU (Preferred)** | NVIDIA GPU (4GB+ VRAM) | 10–30 seconds per image | Uses CUDA + `torch.float16`        |
| **CPU (Fallback)**  | 8GB+ RAM               | 2–5 minutes per image   | Uses CPU + `torch.float32`         |

The `ImageGenerator` class:

* Automatically detects GPU/CPU.
* Adjusts precision.
* Handles optional performance libraries like `xformers` gracefully.

---

## **5. Usage Instructions (With Example Prompts)**

1. Start the app:

   ```bash
   streamlit run app.py
   ```
2. All UI inputs persist through `st.session_state`.
3. Enter your prompt in the main text field.
4. Choose a **Style Preset** (e.g., *Cyberpunk*), which adds advanced prompt tags automatically.
5. Click **Generate Images**.
6. Watch real-time progress indicators.
7. Generated images (with watermark) appear in the UI and are saved to:

   ```
   generated_images/
   ```
8. Export images using:

   * **Download PNG**
   * **Download JPEG**

---

## **6. Prompt Engineering Tips**

The app includes Style Presets, but users can further optimize prompts using:

### **Clarity**

Define subject, scene, and action precisely.

### **Detail & Quality**

Include descriptors like:

* `4K`,
* `highly detailed`,
* `octane render`,
* `cinematic lighting`.

### **Negative Prompts**

Always specify undesired elements, such as:

```
blurry, distorted, low quality, bad anatomy
```

---

## **7. Ethical AI Use**

The project satisfies ethical AI requirements through:

### **1. Content Filtering**

The Stable Diffusion Safety Checker (from Diffusers) automatically blocks harmful or explicit content.

### **2. AI Watermarking**

The `add_watermark` function stamps every output with:

```
AI Generated | Talrn
```

ensuring transparency and responsible use.

### **3. Usage Guidelines**

Users are advised to avoid producing harmful, illegal, or unethical content.

---

## **8. Limitations and Future Improvements**

### **Research Summary: Diffusion Models vs GANs**

Diffusion models (used here) outperform GANs in:

* Photorealism
* Fine detail
* Output diversity
  They reverse a noise process to generate stable, high-quality images.

---

### **Current Limitations**

* **Slow on CPU** (minutes per image)
* **High VRAM requirement** for best results

---

### **Future Enhancements**

#### **1. Fine-Tuning**

Add support for LoRA training on custom datasets for domain-specific image styles.

#### **2. Style Transfer Feature**

Allow users to upload an image and apply new styles through prompt + image conditioning.

#### **3. Optimized Deployment**

Docker containerization + GPU cloud deployment for scalable, hardware-free usage.

---

If you'd like, I can also **format this as a PDF, DOCX, or PowerPoint** using the appropriate tool.
