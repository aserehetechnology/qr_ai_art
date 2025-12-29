# ✨ Smart QR AI Art Generator v2.0

Transform functional QR codes into stunning AI-generated artwork using Stable Diffusion ControlNet.
Powered by a **Smart Adaptive Engine** that automatically tunes parameters for "Smooth" subjects (faces, cars) vs "Textured" subjects (jungles, ruins).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aserehetechnology/qr_ai_art/blob/main/notebooks/QR_AI_Art_Colab.ipynb)

## Features
- **Smart Auto Mode:** Automatically detects subject type and prevents distortion.
- **Pure AI:** No pixel overlays. The QR code *is* the image.
- **Free GPU Hosting:** Run for free on Google Colab (Link above).
- **ControlNet 1.1:** State-of-the-art QR Code Monster model.

## Quick Start (Free GPU)
1. Click the **Open in Colab** badge above.
2. In Colab, click **Runtime -> Change runtime type** -> Select **T4 GPU**.
3. Click the Play button ▶️ inside the notebook.
4. Wait for the public URL (e.g., `trycloudflare.com`).
5. Generate Art! 🎨

## Local Installation
Requirements: Python 3.10+, NVIDIA GPU (8GB+ VRAM recommended).
```bash
git clone https://github.com/aserehetechnology/qr_ai_art.git
cd qr_ai_art
pip install -r requirements.txt
python web_app.py
```
