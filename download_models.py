from huggingface_hub import snapshot_download
import os

# Enable HF Transfer for high-speed download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Define local model directory
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"🚀 Starting Optimized Model Download to: {MODEL_DIR}")
print("⚡️ Using hf_transfer for max speed + fp16 variants for smaller size")

# 1. Download ControlNet QR Code Monster
# ~1.45GB (Single file)
print("\n[1/2] Downloading ControlNet (QR Code Monster)...")
try:
    cn_path = snapshot_download(
        repo_id="monster-labs/control_v1p_sd15_qrcode_monster",
        local_dir=os.path.join(MODEL_DIR, "control_v1p_sd15_qrcode_monster"),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin", "*.pt", "*.ckpt", "*.h5", "v2/*"], # Only need main safetensors
        resume_download=True
    )
    print(f"✅ ControlNet downloaded to: {cn_path}")
except Exception as e:
    print(f"❌ Error downloading ControlNet: {e}")

# 2. Download Realistic Vision V5.1 (Photorealistic Model)
# Much better than standard SD 1.5 for realism
print("\n[2/2] Downloading Realistic Vision V5.1...")
try:
    sd_path = snapshot_download(
        repo_id="SG161222/Realistic_Vision_V5.1_noVAE",
        local_dir=os.path.join(MODEL_DIR, "stable-diffusion-v1-5"), # Keep dir name same to avoid code changes
        local_dir_use_symlinks=False,
        ignore_patterns=["*.ckpt", "*.h5", "safety_checker/*"],
        resume_download=True
    )
    print(f"✅ Realistic Vision downloaded to: {sd_path}")
except Exception as e:
    print(f"❌ Error downloading Realistic Vision: {e}")

print("\n🎉 All downloads finished! Restart your server to use the models.")
