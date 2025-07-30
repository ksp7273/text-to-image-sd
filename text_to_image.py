import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datasets import load_dataset
import gradio as gr
from torch.cuda.amp import autocast
import os

# 1. Load pre-trained Stable Diffusion model from Hugging Face
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_auth_token=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

# 2. Fine-tune model on a niche dataset (placeholder for Danbooru)
def fine_tune_model(pipe):
    # Placeholder: Fine-tuning requires a GPU and dataset preprocessing
    print("Fine-tuning skipped in Codespaces due to resource constraints.")
    # To fine-tune, uncomment and configure with actual dataset (e.g., Danbooru)
    """
    dataset = load_dataset("hf-internal-testing/diffusers-images", split="train")
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-5,
        fp16=True if torch.cuda.is_available() else False,
        save_steps=500,
        save_total_limit=2
    )
    print("Implement dataset preprocessing and training loop for fine-tuning")
    """
    return pipe

# 3. Gradio interface for user prompts
def generate_image(prompt, num_inference_steps=50, guidance_scale=7.5):
    pipe = load_model()
    pipe = fine_tune_model(pipe)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with autocast() if device == "cuda" else torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=guidance_scale
        ).images[0]
    return image

# Main function to set up and run the app
def main():
    # Define Gradio interface
    interface = gr.Interface(
        fn=generate_image,
        inputs=[
            gr.Textbox(label="Enter your prompt", placeholder="e.g., 'A vibrant anime-style sunset over a futuristic city'"),
            gr.Slider(minimum=10, maximum=100, value=50, label="Inference Steps"),
            gr.Slider(minimum=1, maximum=20, value=7.5, label="Guidance Scale")
        ],
        outputs=gr.Image(type="pil"),
        title="Text-to-Image Generator with Stable Diffusion",
        description="Generate custom artwork from text prompts using Stable Diffusion."
    )
    
    # Launch Gradio interface on Codespaces port
    interface.launch(server_name="0.0.0.0", server_port=8000, share=False)

if __name__ == "__main__":
    main()

# Deployment note:
# To deploy on Azure via GitHub Actions:
# 1. Push this code to your GitHub repository.
# 2. Create a requirements.txt with: torch, diffusers, transformers, datasets, gradio, accelerate, scipy
# 3. Set up a GitHub Actions workflow (see below).
# 4. Configure Azure App Service with GPU support for deployment.