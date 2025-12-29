import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from PIL import Image
import os
import gc

# Detect device (Prioritize MPS for Mac)
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🚀 Using MPS (Metal Performance Shaders) acceleration for Mac")
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class AI_Generator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AI_Generator, cls).__new__(cls)
            cls._instance.pipe = None
            cls._instance.current_mode = None
        return cls._instance

    def load_model(self, mode="balanced"):
        # If model is already loaded in the correct mode, skip
        if self.pipe is not None and self.current_mode == mode:
            return
        
        # If switching modes, force reload
        if self.pipe is not None:
            print(f"🔄 Switching mode from {self.current_mode} to {mode}. Reloading...")
            del self.pipe
            self.pipe = None
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
        
        self.current_mode = mode
        
        # Cleanup memory before loading
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

        print(f"Loading AI Models in [{mode.upper()}] mode...")
        
        # 1. Load ControlNet
        # USE FLOAT32 FOR STABILITY ON MAC/MPS
        # MPS has issues with mixed precision (float16) in some layers
        dtype_to_use = torch.float32 
        
        cn_path = os.path.join(os.getcwd(), "models", "control_v1p_sd15_qrcode_monster")
        if not os.path.exists(cn_path):
             cn_path = "monster-labs/control_v1p_sd15_qrcode_monster" # Fallback to online
             
        controlnet = ControlNetModel.from_pretrained(
            cn_path, 
            torch_dtype=dtype_to_use,
            use_safetensors=True
        )
        
        # 2. Load Stable Diffusion
        # PRIORITIZE Single File Checkpoint (SafeTensor) if available
        model_dir = os.path.join(os.getcwd(), "models")
        sd_path = os.path.join(model_dir, "stable-diffusion-v1-5")
        
        single_file_path = None
        # Look for any .safetensors in models/ that is NOT controlnet
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith(".safetensors") and "control" not in f:
                    single_file_path = os.path.join(model_dir, f)
                    print(f"Found single-file model: {single_file_path}")
                    break
        
        if single_file_path:
            # Load from single file (Highest Priority - Root)
            print(f"Loading from Single File: {single_file_path}")
            self.pipe = StableDiffusionControlNetPipeline.from_single_file(
                single_file_path,
                controlnet=controlnet,
                torch_dtype=dtype_to_use,
                safety_checker=None,
                use_safetensors=True
            )
        else:
            # Check if sd_path (subfolder) contains a single safetensors file
            # This happens when downloading Realistic Vision via snapshot_download
            sub_single_file = None
            if os.path.exists(sd_path):
                 for f in os.listdir(sd_path):
                     if f.endswith(".safetensors") and "control" not in f:
                         sub_single_file = os.path.join(sd_path, f)
                         break
            
            if sub_single_file:
                 print(f"Loading from Sub-folder Single File: {sub_single_file}")
                 self.pipe = StableDiffusionControlNetPipeline.from_single_file(
                    sub_single_file,
                    controlnet=controlnet,
                    torch_dtype=dtype_to_use,
                    safety_checker=None,
                    use_safetensors=True
                )
            else:
                # Fallback to standard Diffusers folder or online
                print("Single file not found, checking folder/online...")
                
                # Check if local folder has actual Diffusers weights
                has_local_weights = False
                if os.path.exists(sd_path):
                    # Check for bin or safetensors in unet folder
                    unet_path = os.path.join(sd_path, "unet")
                    if os.path.exists(unet_path):
                        if any(f.endswith((".bin", ".safetensors")) for f in os.listdir(unet_path)):
                            has_local_weights = True
            
            if has_local_weights:
                model_id_or_path = sd_path
                variant = None 
            else:
                model_id_or_path = "runwayml/stable-diffusion-v1-5"
                variant = "fp16"

            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id_or_path,
                controlnet=controlnet,
                torch_dtype=dtype_to_use,
                safety_checker=None,
                use_safetensors=True,
                variant=variant
            )
        
        # Use DPM++ 2M Karras scheduler for better quality at low steps
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
        
        # --- MODE CONFIGURATION ---
        self.pipe.enable_attention_slicing() # Always enable slicing to save RAM
        
        if mode == "eco":
            # ECO MODE: Slow but Cool (CPU Offloading)
            if DEVICE == "mps":
                try:
                    self.pipe.enable_model_cpu_offload()
                    print("✅ ECO MODE: Enabled Model CPU Offloading (Cooler, Slower)")
                except Exception as e:
                    print(f"⚠️ Eco mode failed: {e}")
                    self.pipe.to(DEVICE)
            else:
                 self.pipe.to(DEVICE)
                 
        else: # balanced
            # BALANCED MODE: Fast & Safe (Full GPU + VAE Tiling)
            self.pipe.to(DEVICE)
            if DEVICE == "mps":
                try:
                    self.pipe.enable_vae_tiling()
                    print("✅ BALANCED MODE: Enabled VAE Tiling (Fast, Crash-Free)")
                except Exception as e:
                    print(f"⚠️ VAE Tiling failed: {e}")

        print("AI Models Loaded Successfully!")

    def generate(self, 
                 control_image: Image.Image, 
                 prompt: str, 
                 negative_prompt: str = "", 
                 controlnet_conditioning_scale: float = 1.35, 
                 guidance_scale: float = 7.0,
                 control_guidance_start: float = 0.0,
                 control_guidance_end: float = 1.0,
                 num_inference_steps: int = 10,
                 seed: int = -1,
                 mode: str = "balanced",
                 callback=None):
        
        # Cleanup memory before generation
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

        self.load_model(mode=mode)
        
        if seed != -1:
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
        else:
            generator = None
        
        # Ensure control image is appropriate size
        # Reduced to 512x512 to prevent Mac overheating/hang
        width, height = 512, 512
        control_image = control_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Run generation
        try:
            with torch.no_grad():
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback=callback,
                    callback_steps=1
                )
            
            # Manual post-processing to avoid black image/NaN issues on MPS
            image_tensor = output.images[0] 
            
            # Cleanup memory after generation
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

            if isinstance(image_tensor, Image.Image):
                return image_tensor
                
            # If output is tensor (sometimes happens with raw output)
            return image_tensor
        except Exception as e:
            # Emergency cleanup if crash
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()
            raise e

# Singleton instance
ai_engine = AI_Generator()
