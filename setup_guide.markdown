# Text-to-Image Generation Setup Guide

This guide provides a step-by-step process to develop and deploy a text-to-image generation tool using Stable Diffusion in GitHub Codespaces, with deployment to Azure via GitHub Actions.

## Prerequisites
- GitHub account with Codespaces access (GitHub Pro for students if needed).
- Azure account with billing enabled.
- Basic knowledge of Python and Git.

## Steps

1. **Create a GitHub Repository**
   - Sign in to GitHub and create a repository (e.g., `text-to-image-sd`).
   - Initialize with a README and a Python `.gitignore`.

2. **Launch GitHub Codespaces**
   - Navigate to the repository, click **Code** > **Codespaces** > **New with options**.
   - Select a machine (e.g., 8-core, 16GB RAM, 64GB storage; GPU-enabled if available).
   - Open the Codespace in browser-based VS Code.

3. **Set Up the Development Environment**
   - Open the terminal (Hamburger menu > **Terminal** > **New Terminal**).
   - Install system dependencies:
     ```bash
     sudo apt-get update
     sudo apt-get install -y libgl1
     ```

4. **Create and Save the Application Code**
   - Create `text_to_image.py` in the repository root with the Stable Diffusion code (includes Gradio interface on port 8000).
   - Ensure it checks for GPU availability (`torch.cuda.is_available()`).

5. **Create `requirements.txt`**
   - Create `requirements.txt` in the repository root:
     ```
     torch
     diffusers
     transformers
     datasets
     gradio
     accelerate
     scipy
     ```

6. **Install Dependencies**
   - Run in the terminal:
     ```bash
     pip install -r requirements.txt
     ```
   - For GPU-enabled Codespaces:
     ```bash
     pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
     pip install -r requirements.txt
     ```
   - Verify with `pip list`.

7. **Test the Application**
   - Run:
     ```bash
     python text_to_image.py
     ```
   - Access the Gradio interface via the Codespaces URL (e.g., `https://<codespace-name>-8000.app.github.dev/`).
   - Test with a prompt (e.g., "A vibrant anime-style sunset over a futuristic city").

8. **Set Up Azure for Deployment**
   - Create an Azure App Service (Linux, GPU-enabled if possible).
   - Download the publish profile from the Azure portal.

9. **Configure GitHub Actions**
   - Create `.github/workflows/deploy.yml` with the deployment workflow.
   - Add the Azure publish profile to GitHub Secrets (`AZURE_WEBAPP_PUBLISH_PROFILE`).

10. **Push Changes to GitHub**
    - Commit and push:
      ```bash
      git add .
      git commit -m "Set up Stable Diffusion project and deployment workflow"
      git push origin main
      ```

11. **Monitor and Access Deployment**
    - Monitor the workflow in the **Actions** tab.
    - Access the app via the Azure URL (e.g., `https://<your-app-name>.azurewebsites.net:8000`).

12. **Post-Deployment Optimization**
    - Configure Azure for GPU support (e.g., NC-series VMs).
    - Add error handling and logging to the code.
    - Secure the Gradio interface with authentication.

## Notes
- **GPU Support**: GPU-enabled Codespaces are in private preview; CPU-based inference is slower.
- **Costs**: Ensure billing for GitHub Codespaces and Azure.
- **Fine-Tuning**: Skipped in Codespaces due to resource constraints; use Azure Machine Learning for fine-tuning on a dataset like Danbooru.