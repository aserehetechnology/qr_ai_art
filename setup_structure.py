import os
from huggingface_hub import snapshot_download

# Define local model directory
MODEL_DIR = os.path.join(os.getcwd(), "models")

print("🚀 Setting up model folder structure and downloading lightweight configs...")

# 1. Setup ControlNet Structure (Configs only)
print("\n[1/2] Setting up ControlNet (Configs)...")
try:
    snapshot_download(
        repo_id="monster-labs/control_v1p_sd15_qrcode_monster",
        local_dir=os.path.join(MODEL_DIR, "control_v1p_sd15_qrcode_monster"),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.ckpt", "*.h5"], # Ignore heavy files
    )
    print("✅ ControlNet configs ready.")
except Exception as e:
    print(f"❌ Error setting up ControlNet: {e}")

# 2. Setup Stable Diffusion v1.5 FP16 Structure (Configs only)
print("\n[2/2] Setting up Stable Diffusion v1.5 (Configs)...")
try:
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir=os.path.join(MODEL_DIR, "stable-diffusion-v1-5"),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors", "*.bin", "*.ckpt", "*.h5"], # Ignore heavy files
    )
    print("✅ Stable Diffusion configs ready.")
except Exception as e:
    print(f"❌ Error setting up Stable Diffusion: {e}")

print("\nPLEASE MANUALLY DOWNLOAD THESE FILES AND PLACE THEM IN THE FOLDERS:")
print(f"Base Directory: {MODEL_DIR}")
