from __future__ import annotations

import base64
import io
import hashlib
import threading
import uuid
import time
import json
from dataclasses import dataclass

from flask import Flask, request, jsonify, render_template_string
from PIL import Image

from qr_ai_art import Style, _clamp01, _parse_color, generate_art_qr, create_finder_mask
from PIL import ImageEnhance, ImageChops, ImageFilter

# Simple in-memory cache for generated results
# Key: Hash of form parameters
# Value: Base64 string of the generated image
RESULT_CACHE = {}

# Async Task Store
# Key: task_id
# Value: { 'status': 'pending'|'processing'|'completed'|'failed'|'cancelled', 'progress': 0, 'step': 0, 'total': 0, 'result': None, 'error': None }
TASKS = {}

# --- SMART PROMPT ANALYZER ---
def smart_analyze_prompt(prompt_text):
    """
    Analyzes the prompt to decide the best strategy:
    1. Smooth/Geometric Subjects -> Hybrid Mode (Structure + Micro-Blend)
       (Prevents "dents" on faces, cars, text, smooth surfaces)
    2. Textured/Organic Subjects -> Shadow Art Mode (Pure Structure)
       (Hides QR perfectly in complexity)
    """
    p = prompt_text.lower()
    
    # KATEGORI 1: HARUS MULUS (Risiko tinggi jika dipaksa ControlNet)
    # Jika ada salah satu kata ini, kita aktifkan Blending 15% agar aman.
    smooth_keywords = [
        # Vehicles
        "truck", "car", "bus", "train", "plane", "ship", "boat", "yacht", "bike", 
        "motorcycle", "scooter", "spaceship", "robot", "mech", "cyborg", "machine",
        
        # People & Characters
        "face", "portrait", "girl", "boy", "man", "woman", "person", "character", 
        "anime", "manga", "waifu", "cartoon", "goddess", "warrior", "elf", "princess",
        "king", "queen", "knight", "human", "body", "skin", "hair",
        
        # Styles & Materials
        "vector", "illustration", "flat", "minimalist", "clean", "simple", "3d render", 
        "smooth", "shiny", "glass", "metal", "plastic", "chrome", "polished", "neon",
        "gradient", "studio lighting", "soft", "bokeh", "macro", "unreal engine", "blender",
        
        # Elements
        "sky", "cloud", "water", "sea", "ocean", "beach", "ice", "snow field", "white background",
        "space", "galaxy", "stars", "moon", "sun", "sunset", "sunrise"
    ]
    
    # KATEGORI 2: TEKSTUR KASAR (Sangat cocok untuk Shadow QR)
    # Ini bisa menelan QR code tanpa blending.
    texture_keywords = [
        # Nature
        "jungle", "forest", "tree", "plant", "flower", "leaf", "grass", "garden", "park",
        "mountain", "rock", "cliff", "canyon", "cave", "desert", "sand", "waterfall", 
        "volcano", "lava", "fire", "smoke",
        
        # Architecture
        "ruins", "ancient", "temple", "castle", "brick", "stone", "wall", "city", 
        "skyscraper", "building", "house", "village", "street", "bridge", "aerial", "map",
        "library", "books", "factory", "industrial",
        
        # Patterns & Art
        "texture", "pattern", "mosaic", "stained glass", "circuit", "cyberpunk", 
        "steampunk", "intricate", "detailed", "painting", "oil", "sketch", "drawing",
        "graffiti", "grunge", "rust", "old", "dirty"
    ]
    
    is_smooth = any(k in p for k in smooth_keywords)
    is_textured = any(k in p for k in texture_keywords)
    
    # LOGIC PRIORITAS:
    # Keselamatan nomor 1. Jika ada objek halus (misal: "anime girl in jungle"), 
    # kita tetap pilih Mode Halus agar wajahnya tidak rusak.
    
    if is_smooth:
        # HYBRID MODE: Structure 1.70 + Blend 0.15
        # Aman untuk wajah, kendaraan, dan benda licin.
        return {
            "mode": "Smooth/Adaptive",
            "cn_scale": 1.70,      
            "blend": 0.15,         
            "contrast": 1.05,
            "sharpness": 1.15
        }
        
    elif is_textured:
        # SHADOW MODE: Structure 1.65 + Blend 0.00
        # Target: "Perfect Illusion" (Lush Forest Reference)
        # Reduced from 1.85 to 1.65 to allow leaves to grow naturally over the grid
        return {
            "mode": "Textured/Nature",
            "cn_scale": 1.65,      
            "blend": 0.00,         
            "contrast": 1.20,
            "sharpness": 1.50
        }
        
    else:
        # BALANCED MDOE
        # Fallback jika prompt tidak spesifik
        return {
            "mode": "Balanced/General",
            "cn_scale": 1.75,
            "blend": 0.10,         
            "contrast": 1.10,
            "sharpness": 1.25
        }

def blend_qr_contrast(ai_image, control_image, qr_data, opacity=0.35):
    """
    Subtly enforces the QR structure by burning shadows and dodging highlights.
    Opacity controls the strength of the enforcement.
    """
    # 1. Resize control to match AI image
    control = control_image.resize(ai_image.size, Image.Resampling.NEAREST).convert("L")
    
    # 2. Create Shadow Layer (Darken original)
    # Stronger effect = darker shadows
    shadow_factor = 0.4 - (opacity * 0.2) # 0.4 to 0.2
    shadow_layer = ImageEnhance.Brightness(ai_image).enhance(max(0.1, shadow_factor))
    
    # 3. Create Highlight Layer (Brighten original)
    # Stronger effect = brighter highlights
    highlight_factor = 1.0 + (opacity * 0.8) # 1.0 to 1.8
    highlight_layer = ImageEnhance.Brightness(ai_image).enhance(highlight_factor)
    
    # 4. Composite Global (Data Modules)
    # Mask 0 (Black) -> Shadow. Mask 255 (White) -> Highlight.
    global_contrast = Image.composite(highlight_layer, shadow_layer, control)
    
    # 5. Blend Global: Apply opacity to data
    blended = Image.blend(ai_image, global_contrast, opacity)
    
    # 6. Special Handling for Finder Patterns (The 3 big squares)
    try:
        w, h = ai_image.size
        # CRITICAL FIX: Border must match the generator's border (0)
        # Otherwise the blend mask will be misaligned with the AI structure!
        finder_img = create_finder_mask(qr_data, size=w, border=0)
        finder_alpha = finder_img.split()[3]
        
        # Blur the alpha mask to avoid hard edges
        finder_alpha = finder_alpha.filter(ImageFilter.GaussianBlur(radius=4))
        
        finder_pattern = finder_img.convert("L")
        
        # Finder Contrast: Usually stronger than global
        deep_shadow = ImageEnhance.Brightness(ai_image).enhance(max(0.05, shadow_factor - 0.1))
        bright_highlight = ImageEnhance.Brightness(ai_image).enhance(highlight_factor + 0.2)
        
        finder_contrast = Image.composite(bright_highlight, deep_shadow, finder_pattern)
        
        # Blend finders slightly stronger than global, BUT respect the base opacity!
        # If opacity is 0.0 (Hidden), finder opacity should also be very low/zero
        # to avoid ugly box artifacts. We trust the ControlNet 1.95 to structure it instead.
        finder_opacity = min(1.0, opacity + 0.15) if opacity > 0 else 0.0
        
        final_finders = Image.blend(blended, finder_contrast, finder_opacity)
        
        final_output = Image.composite(final_finders, blended, finder_alpha)
        
        return final_output
        
    except Exception as e:
        print(f"Finder enhancement failed: {e}")
        return blended


@dataclass(frozen=True)
class FormValues:
    data: str
    size: int
    border: int
    dark: str
    light: str
    dark_alpha: float
    light_alpha: float
    rounded: int
    preserve_finders: bool
    strength: float
    texture: float
    mode: str
    readability: str # 'hidden', 'balanced', 'scannable'
    guidance_scale: float
    control_end: float


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


_PAGE = """<!doctype html>
<html lang="id">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>QR Art Generator</title>
    <style>
      :root { 
        --bg: #0b1020; 
        --card: #121a33; 
        --text: #eaf0ff; 
        --muted: #a8b3d6; 
        --line: rgba(255,255,255,.08); 
        --accent: #4f7cff;
        --success: #10b981;
        --warning: #f59e0b;
      }
      
      body { 
        margin: 0; 
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; 
        background: radial-gradient(1200px 600px at 50% 0%, #182553, var(--bg)); 
        color: var(--text); 
      }
      
      .wrap { max-width: 1120px; margin: 0 auto; padding: 32px 20px 60px; }
      
      h1 { font-size: 28px; margin: 0 0 6px; letter-spacing: -.3px; font-weight: 700; }
      .sub { font-size: 14px; color: var(--muted); margin: 0 0 28px; opacity: 0.85; }
      
      .grid { display: grid; grid-template-columns: 440px 1fr; gap: 20px; }
      @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
      
      .card { 
        background: color-mix(in srgb, var(--card) 92%, black); 
        border: 1px solid var(--line); 
        border-radius: 16px; 
        padding: 20px; 
      }
      
      /* SECTION HEADERS */
      .section-header {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--accent);
        margin: 24px 0 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(79, 124, 255, 0.2);
      }
      .section-header:first-child { margin-top: 0; }
      
      /* FORM ELEMENTS */
      label { 
        display: block; 
        font-size: 13px; 
        font-weight: 600;
        color: var(--text); 
        margin: 14px 0 8px; 
      }
      
      input[type="text"], input[type="number"], select, textarea { 
        width: 100%; 
        box-sizing: border-box; 
        padding: 11px 12px; 
        border-radius: 10px; 
        border: 1px solid var(--line); 
        background: rgba(0,0,0,.25); 
        color: var(--text); 
        outline: none; 
        font-size: 14px; 
        font-family: inherit; 
        transition: border-color 0.2s, background 0.2s;
      }
      
      input:focus, select:focus, textarea:focus {
        border-color: var(--accent);
        background: rgba(0,0,0,.35);
      }
      
      textarea { resize: vertical; min-height: 80px; }
      input[type="file"] { width: 100%; color: var(--muted); font-size: 12px; }
      
      .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
      
      .hint { 
        font-size: 11px; 
        color: var(--muted); 
        margin-top: 6px; 
        line-height: 1.5; 
        opacity: 0.75; 
      }
      
      /* BUTTONS */
      .btn { 
        margin-top: 20px; 
        width: 100%; 
        padding: 14px 16px; 
        background: var(--accent); 
        border: 0; 
        color: white; 
        border-radius: 11px; 
        font-weight: 700; 
        font-size: 15px;
        cursor: pointer; 
        transition: 0.2s; 
        box-shadow: 0 4px 12px rgba(79, 124, 255, 0.3);
      }
      .btn:hover { 
        filter: brightness(1.15); 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(79, 124, 255, 0.4);
      }
      
      /* STYLE SELECTOR CARDS */
      .style-selector {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 12px 0 20px;
      }
      
      .style-card {
        position: relative;
        padding: 16px 12px;
        background: rgba(0,0,0,.2);
        border: 2px solid var(--line);
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.25s ease;
        text-align: center;
      }
      
      .style-card:hover {
        background: rgba(79, 124, 255, 0.08);
        border-color: rgba(79, 124, 255, 0.4);
        transform: translateY(-2px);
      }
      
      .style-card.active {
        background: rgba(79, 124, 255, 0.15);
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(79, 124, 255, 0.2);
      }
      
      .style-card .icon {
        font-size: 32px;
        margin-bottom: 8px;
        display: block;
      }
      
      .style-card .title {
        font-size: 13px;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 4px;
      }
      
      .style-card .desc {
        font-size: 10px;
        color: var(--muted);
        line-height: 1.3;
      }
      
      .style-card input[type="radio"] {
        position: absolute;
        opacity: 0;
        pointer-events: none;
      }
      
      /* TEMPLATE GRID */
      .template-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin: 12px 0;
      }
      
      .template-item {
        padding: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--line);
        border-radius: 10px;
        cursor: pointer;
        transition: 0.2s;
        font-size: 12px;
        text-align: center;
      }
      
      .template-item:hover {
        background: rgba(79, 124, 255, 0.1);
        border-color: var(--accent);
      }
      
      .template-item.selected {
        background: rgba(79, 124, 255, 0.2);
        border-color: var(--accent);
      }
      
      /* PRESETS */
      .presets { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 16px; }
      .pset { 
        padding: 10px; 
        font-size: 11px; 
        background: rgba(255,255,255,0.04); 
        border: 1px solid var(--line); 
        border-radius: 10px; 
        color: var(--muted); 
        cursor: pointer; 
        text-align: center; 
        transition: 0.2s; 
        font-weight: 600;
      }
      .pset:hover { 
        background: rgba(255,255,255,0.1); 
        color: var(--text); 
        border-color: var(--accent); 
      }
      
      /* ERROR & SUCCESS MESSAGES */
      .error { 
        background: rgba(255, 93, 122, .12); 
        border: 1px solid rgba(255, 93, 122, .35); 
        color: #ffd2dc; 
        padding: 12px 14px; 
        border-radius: 12px; 
        margin: 12px 0 0; 
        font-size: 13px; 
      }
      
      /* PREVIEW */
      .preview { 
        display: flex; 
        flex-direction: column; 
        gap: 16px; 
        align-items: center; 
        justify-content: center; 
        min-height: 480px; 
      }
      
      .imgbox { 
        background: rgba(255,255,255,.035); 
        border: 2px dashed var(--line); 
        border-radius: 16px; 
        padding: 20px; 
        width: 100%; 
        box-sizing: border-box; 
        display: grid; 
        place-items: center; 
        min-height: 400px;
      }
      
      .imgbox img { 
        max-width: min(560px, 100%); 
        height: auto; 
        border-radius: 12px; 
        box-shadow: 0 12px 48px -12px rgba(0,0,0,0.6); 
      }
      
      .dl { 
        display: inline-block; 
        padding: 10px 16px; 
        border: 1px solid var(--line); 
        border-radius: 10px; 
        color: var(--text); 
        text-decoration: none; 
        background: rgba(0,0,0,.25); 
        font-size: 13px; 
        font-weight: 600;
        transition: 0.2s;
      }
      .dl:hover { 
        filter: brightness(1.15); 
        border-color: var(--accent);
      }
      
      .foot { 
        margin-top: 16px; 
        font-size: 12px; 
        color: var(--muted); 
        text-align: center; 
        line-height: 1.6;
      }
      
      /* MODE TOGGLE */
      .mode-toggle { 
        display: flex; 
        background: rgba(0,0,0,0.25); 
        border-radius: 12px; 
        padding: 5px; 
        margin-bottom: 24px; 
        border: 1px solid var(--line);
      }
      
      .mode-btn { 
        flex: 1; 
        border: none; 
        background: transparent; 
        color: var(--muted); 
        padding: 10px; 
        border-radius: 9px; 
        cursor: pointer; 
        font-size: 13px; 
        font-weight: 700; 
        transition: 0.2s; 
      }
      
      .mode-btn.active { 
        background: var(--accent); 
        color: white; 
        box-shadow: 0 2px 8px rgba(79, 124, 255, 0.3);
      }
      
      /* AI CONTROLS */
      #ai-controls { 
        display: none; 
      }
      
      /* COLLAPSIBLE SECTIONS */
      .collapsible {
        margin-top: 20px;
        border-top: 1px solid var(--line);
        padding-top: 16px;
      }
      
      .collapsible-header {
        cursor: pointer;
        color: var(--text);
        font-weight: 700;
        font-size: 13px;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 0;
        user-select: none;
      }
      
      .collapsible-header:hover {
        color: var(--accent);
      }
      
      .collapsible-content {
        margin-top: 12px;
      }
      
      /* LOADING OVERLAY */
      #loading-overlay {
        display: none;
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(11, 16, 32, 0.97);
        z-index: 9999;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(8px);
      }
      
      .spinner {
        width: 64px; height: 64px;
        border: 5px solid rgba(255,255,255,0.1);
        border-top: 5px solid var(--accent);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 24px;
      }
      
      @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      
      .loading-text { 
        font-size: 20px; 
        font-weight: 700; 
        color: white; 
        letter-spacing: 0.3px; 
      }
      
      .loading-sub { 
        font-size: 14px; 
        color: var(--muted); 
        margin-top: 10px; 
      }
      
      .progress-container { 
        width: 320px; 
        height: 8px; 
        background: rgba(255,255,255,0.1); 
        border-radius: 4px; 
        margin-top: 20px; 
        overflow: hidden; 
      }
      
      .progress-bar { 
        height: 100%; 
        background: linear-gradient(90deg, var(--accent), #6b8fff); 
        width: 0%; 
        transition: width 0.3s ease; 
      }
      
      .cancel-btn { 
        margin-top: 24px; 
        padding: 10px 20px; 
        background: transparent; 
        border: 2px solid rgba(255, 93, 122, .5); 
        color: #ff5d7a; 
        border-radius: 10px; 
        cursor: pointer; 
        font-size: 13px; 
        font-weight: 600;
        transition: 0.2s; 
      }
      
      .cancel-btn:hover { 
        background: rgba(255, 93, 122, .15); 
        border-color: #ff5d7a;
      }
      
      /* DEMO BANNER */
      .demo-banner {
        background: linear-gradient(135deg, rgba(79, 124, 255, 0.15), rgba(16, 185, 129, 0.15));
        border: 1px solid rgba(79, 124, 255, 0.3);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
      }
      
      .demo-banner .icon {
        font-size: 24px;
      }
      
      .demo-banner .content {
        flex: 1;
      }
      
      .demo-banner .title {
        font-size: 13px;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 2px;
      }
      
      .demo-banner .desc {
        font-size: 11px;
        color: var(--muted);
      }
    </style>
    <script>
        let currentTaskId = null;
        let pollInterval = null;

        function setMode(mode) {
            document.getElementById('mode-input').value = mode;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('btn-' + mode).classList.add('active');
            
            if(mode === 'ai') {
                document.getElementById('ai-controls').style.display = 'block';
                document.getElementById('classic-controls').style.display = 'none';
                document.querySelector('.presets').style.display = 'none';
            } else {
                document.getElementById('ai-controls').style.display = 'none';
                document.getElementById('classic-controls').style.display = 'block';
                document.querySelector('.presets').style.display = 'grid';
            }
        }
        
        async function submitForm(event) {
            const mode = document.getElementById('mode-input').value;
            
            if (mode === 'ai') {
                event.preventDefault(); // Stop normal submit
                
                // Show overlay
                document.getElementById('loading-overlay').style.display = 'flex';
                document.getElementById('loading-status').innerText = "Initializing AI...";
                document.getElementById('loading-sub').innerText = "Starting engine...";
                document.getElementById('progress-bar').style.width = '0%';
                document.getElementById('cancel-btn').style.display = 'block';
                document.getElementById('classic-spinner').style.display = 'none'; // Hide spinner for progress bar focus
                document.getElementById('progress-container').style.display = 'block';

                // Set random seed if empty or just rely on backend? 
                // Let's force a random seed in the form before sending if not set
                if(!document.querySelector('[name=seed]').value) {
                     document.querySelector('[name=seed]').value = Math.floor(Math.random() * 1000000000);
                }
                
                const form = event.target;
                const formData = new FormData(form);
                
                try {
                    // Start Generation
                    const res = await fetch('/generate_ai', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    
                    if (data.error) throw new Error(data.error);
                    
                    currentTaskId = data.task_id;
                    pollProgress(currentTaskId);
                    
                } catch (e) {
                    alert("Error: " + e.message);
                    hideLoading();
                }
            } else {
                // Classic mode: let it submit normally
                showLoading();
            }
        }

        async function pollProgress(taskId) {
            if(pollInterval) clearInterval(pollInterval);
            
            pollInterval = setInterval(async () => {
                try {
                    const res = await fetch('/progress/' + taskId);
                    const data = await res.json();
                    
                    if (data.status === 'completed') {
                        clearInterval(pollInterval);
                        // Update Image
                        const imgBox = document.querySelector('.imgbox');
                        imgBox.innerHTML = `<img src="data:image/png;base64,${data.result}" alt="Result" />`;
                        
                        // Add Download Button (Fix for missing button)
                        // Remove existing download buttons if any to avoid duplicates
                        const existingBtn = document.getElementById('dl-btn-container');
                        if(existingBtn) existingBtn.remove();

                        const dlDiv = document.createElement('div');
                        dlDiv.id = 'dl-btn-container';
                        dlDiv.style.cssText = "display:flex; gap:10px; justify-content:center; margin-top:12px;";
                        dlDiv.innerHTML = `<a href="data:image/png;base64,${data.result}" download="qr_art_${data.task_id}.png" class="dl">⬇️ Download PNG</a>`;
                        
                        // Append after imgbox
                        imgBox.parentNode.appendChild(dlDiv);
                        
                        hideLoading();
                    } else if (data.status === 'failed') {
                        clearInterval(pollInterval);
                        alert("Generation Failed: " + data.error);
                        hideLoading();
                    } else if (data.status === 'cancelled') {
                         clearInterval(pollInterval);
                         hideLoading();
                    } else {
                        // Update Progress
                        const pct = data.progress + '%';
                        document.getElementById('progress-bar').style.width = pct;
                        document.getElementById('loading-status').innerText = `Generating Step ${data.step}/${data.total}`;
                        document.getElementById('loading-sub').innerText = `${pct} Complete`;
                    }
                } catch (e) {
                    console.error(e);
                }
            }, 1000);
        }

        async function cancelGeneration() {
            if (currentTaskId) {
                await fetch('/cancel/' + currentTaskId, { method: 'POST' });
                document.getElementById('loading-status').innerText = "Cancelling...";
                document.getElementById('cancel-btn').style.display = 'none';
            }
        }

        function hideLoading() {
            document.getElementById('loading-overlay').style.display = 'none';
            if(pollInterval) clearInterval(pollInterval);
        }

        function showLoading() {
            // Set random seed for every new generation to ensure variety
            document.getElementById('seed-input').value = Math.floor(Math.random() * 1000000000);

            const overlay = document.getElementById('loading-overlay');
            const status = document.getElementById('loading-status');
            const sub = document.getElementById('loading-sub');
            
            overlay.style.display = 'flex';
            document.getElementById('classic-spinner').style.display = 'block';
            document.getElementById('progress-container').style.display = 'none';
            document.getElementById('cancel-btn').style.display = 'none';
            
            const messages = [
                "🎨 Mixing colors...",
                "🧠 Igniting neural networks...",
                "🌲 Planting digital forests...",
                "🏗️ Constructing QR architecture...",
                "✨ Adding magic dust...",
                "📸 Adjusting focus...",
                "🖼️ Rendering final masterpiece..."
            ];
            
            let i = 0;
            setInterval(() => {
                status.innerText = messages[i % messages.length];
                i++;
            }, 2500);
        }
        
        function setPreset(name) {
            const sets = {
                'forest': { s: 1.1, t: 1.0, da: 0.50, la: 0.10, m: 'organic' },
                'city': { s: 0.9, t: 0.6, da: 0.60, la: 0.15, m: 'organic' },
                'sharp': { s: 1.0, t: 0.3, da: 0.62, la: 0.18, m: 'sharp' }
            };
            const p = sets[name];
            if(!p) return;
            
            setMode('organic');
            document.querySelector('[name=strength]').value = p.s;
            document.querySelector('[name=texture]').value = p.t;
            document.querySelector('[name=dark_alpha]').value = p.da;
            document.querySelector('[name=light_alpha]').value = p.la;
            document.querySelector('[name=mode]').value = p.m;
        }

        const AI_TEMPLATES = {
            "nature": "ancient ruins with moss aerial view, simple vegetation",
            "hidden_jungle": "jungle aerial view, palm trees, green leaves, natural",
            "snow_village": "snow village aerial view, white roofs, simple houses",
            "cyberpunk": "cyberpunk city from above, neon lights, urban grid",
            "steampunk": "mechanical gears pattern, bronze metal, simple",
            "floral": "flower garden aerial view, colorful petals, simple pattern",
            "circuit": "circuit board pattern, blue and black, simple lines",
            "mosaic": "colorful mosaic tiles, simple geometric pattern",
            "liquid": "liquid marble swirls, gold and black, simple",
            "cloud": "white clouds aerial view, blue sky, soft",
            "mountain": "snowy mountains aerial view, peaks and valleys"
        };

        function applyAiTemplate(key) {
            const prompt = AI_TEMPLATES[key];
            if(prompt) {
                document.querySelector('[name=prompt]').value = prompt;
            }
        }

        function randomAiPrompt() {
            const keys = Object.keys(AI_TEMPLATES);
            const randomKey = keys[Math.floor(Math.random() * keys.length)]
;
            applyAiTemplate(randomKey);
        }

        // STYLE CARD SELECTION
        function selectStyle(styleName) {
            // Update radio button
            document.querySelectorAll('.style-card').forEach(card => {
                card.classList.remove('active');
            });
            
            const selectedCard = document.querySelector(`[data-style="${styleName}"]`);
            if (selectedCard) {
                selectedCard.classList.add('active');
                // Also check the radio button
                selectedCard.querySelector('input[type="radio"]').checked = true;
                
                // TRANSPARENCY UPDATE: Auto-set ControlNet Scale based on mode
                // hidden = 1.35 IS THE TRIGGER FOR SMART BACKEND LOGIC
                // If user changes this manually, backend will trust user instead of smart logic.
                const cnInput = document.getElementById('settings-cn-scale');
                if (cnInput) {
                    if (styleName === 'hidden') cnInput.value = "1.35"; // Smart Auto Trigger
                    else if (styleName === 'scannable') cnInput.value = "1.95"; // Max Power
                    else cnInput.value = "1.70"; // Balanced Standard
                }
            }
        }

        // COLLAPSIBLE TOGGLE
        function toggleCollapsible(id) {
            const content = document.getElementById(id);
            const header = content.previousElementSibling;
            const arrow = header.querySelector('.arrow');
            
            if (content.style.display === 'none' || content.style.display === '') {
                content.style.display = 'block';
                if (arrow) arrow.textContent = '▼';
            } else {
                content.style.display = 'none';
                if (arrow) arrow.textContent = '▶';
            }
        }

        // Initialize style cards on load
        window.addEventListener('DOMContentLoaded', function() {
            // Set default active style
            const defaultStyle = document.querySelector('input[name="readability"]:checked');
            if (defaultStyle) {
                const card = defaultStyle.closest('.style-card');
                if (card) {
                    // Trigger click to apply visual state AND cn_scale update
                    // But prevent double radio checking (handled by click) is fine
                    card.click();
                }
            }
        });
    </script>
  </head>
  <body>
    <!-- (Loading overlay skipped) -->
    
    <!-- (Wrap content) -->
    <div id="loading-overlay">
        <div class="spinner" id="classic-spinner"></div>
        <div class="loading-text" id="loading-status">Generating AI Art...</div>
        <div class="loading-sub" id="loading-sub">Creating your masterpiece...<br><span style="font-size:0.8em; opacity:0.7">(Duration depends on your hardware)</span></div>
        
        <div class="progress-container" id="progress-container" style="display:none;">
            <div class="progress-bar" id="progress-bar"></div>
        </div>
        
        <button id="cancel-btn" class="cancel-btn" style="display:none;" onclick="cancelGeneration()">Batal (Cancel)</button>
    </div>

    <div class="wrap">
      <h1>✨ QR AI Art</h1>
      <div class="sub">Transform QR codes into stunning artwork using Advanced AI</div>
      <div class="grid">
        <div class="card">

          <form method="post" action="/generate" enctype="multipart/form-data" onsubmit="submitForm(event)">
            <input type="hidden" name="mode" value="ai">
            <input type="hidden" name="seed" id="seed-input" value="">
          
            <!-- STEP 1: CONTENT -->
            <div class="section-header">1. What connects you?</div>
            <input name="data" type="text" value="{{data}}" placeholder="https://your-website.com" required style="font-size: 16px; padding: 12px;" />
            
            <!-- PROMPT (Visual Description) -->
            <div class="section-header" style="margin-top: 20px;">2. Describe the Art</div>
            <textarea name="prompt" rows="3" placeholder="Example: a futuristic city with neon lights, detailed, 8k" style="font-size: 16px; padding: 12px;">{{prompt}}</textarea>
            <div class="hint">✨ Our Smart AI will automatically detect your subject and optimize the QR code.</div>

            <!-- HIDDEN COMPLEXITY (PRO MODE) -->
            <div class="collapsible" style="margin-top: 30px; border-top: 1px solid var(--line); padding-top: 20px;">
                <div class="collapsible-header" onclick="toggleCollapsible('pro-controls')" style="display: flex; align-items: center; gap: 10px; color: var(--text-dim); cursor: pointer;">
                    <span class="arrow">▶</span> 
                    <span style="font-weight: 500; font-size: 14px;">🛠️ Manual Tuning (Advanced)</span>
                </div>
                
                <div id="pro-controls" class="collapsible-content" style="display: none; padding-top: 15px;">
                    
                    <!-- STYLE SELECTOR (Moved to Pro) -->
                    <label>Scan Strategy Strategy</label>
                    <div class="style-selector">
                        <div class="style-card" data-style="hidden" onclick="selectStyle('hidden')">
                            <input type="radio" name="readability" value="hidden" {% if readability == 'hidden' %}checked{% endif %}>
                            <span class="icon">✨</span>
                            <div class="title">Smart Auto</div>
                            <div class="desc">AI adapts to prompt</div>
                        </div>
                        
                        <div class="style-card active" data-style="balanced" onclick="selectStyle('balanced')">
                            <input type="radio" name="readability" value="balanced" checked>
                            <span class="icon">⚖️</span>
                            <div class="title">Balanced</div>
                            <div class="desc">Standard Control</div>
                        </div>
                        
                        <div class="style-card" data-style="scannable" onclick="selectStyle('scannable')">
                            <input type="radio" name="readability" value="scannable" {% if readability == 'scannable' %}checked{% endif %}>
                            <span class="icon">📱</span>
                            <div class="title">Max Scan</div>
                            <div class="desc">High readability</div>
                        </div>
                    </div>

                    <!-- TEMPLATES -->
                    <div style="margin-top: 15px;">
                         <label>Quick Templates</label>
                         <select onchange="applyAiTemplate(this.value)" style="width:100%; padding:8px; border-radius:8px; background:rgba(0,0,0,.25); border:1px solid var(--line); color:var(--text);">
                            <option value="">-- Choose Template --</option>
                            <option value="hidden_jungle">🌴 Hidden Jungle</option>
                            <option value="snow_village">🏯 Snow Village</option>
                            <option value="nature">🌲 Forest Ruins</option>
                            <option value="mountain">🏔️ Mountains</option>
                            <option value="cloud">☁️ Clouds</option>
                            <option value="cyberpunk">🌃 Cyberpunk</option>
                            <option value="circuit">💻 Circuit Board</option>
                            <option value="liquid">💧 Liquid Art</option>
                        </select>
                    </div>

                    <!-- TECH SLIDERS -->
                    <div style="margin-top: 15px;">
                        <label>Negative Prompt</label>
                        <input name="negative_prompt" type="text" value="{{negative_prompt}}" />
                    
                        <label>QR Visibility Control (CN Scale)</label>
                        <input name="cn_scale" id="settings-cn-scale" type="number" step="0.05" min="0.5" max="2.0" value="1.35" />
                        
                        <label>Performance Mode</label>
                        <select name="performance_mode">
                            <option value="balanced">🚀 Balanced</option>
                            <option value="eco">❄️ Eco Mode</option>
                        </select>
                        
                        <!-- Extra Advanced (CFG etc) -->
                         <div style="margin-top: 15px; border-top: 1px dashed var(--line); padding-top: 10px;">
                            <label>Guidance Scale (CFG)</label>
                            <input name="guidance_scale" type="number" step="0.5" min="1.0" max="20.0" value="7.5" />
                            
                            <label>Control End Point</label>
                            <input name="control_end" type="number" step="0.05" min="0.0" max="1.0" value="1.0" />
                        </div>
                    </div>
                </div>
            </div>

            <!-- CLASSIC MODE -->
            <div id="classic-controls" style="display:none;">
                <div class="section-header">Step 2: Choose Preset</div>
                <div class="presets">
                  <div class="pset" onclick="setPreset('forest')">🌲 Forest</div>
                  <div class="pset" onclick="setPreset('city')">🏙️ City</div>
                  <div class="pset" onclick="setPreset('sharp')">⬛ Sharp</div>
                </div>

                <div class="section-header">Step 3: Upload Background</div>
                <label>Background Image (optional)</label>
                <input name="image" type="file" accept="image/*" />
                <div class="hint">📸 Upload an image or use auto-generated placeholder</div>

                <label>Output Size (pixels)</label>
                <input name="size" type="number" min="256" max="2048" step="64" value="{{size}}" />

                <!-- CLASSIC ADVANCED -->
                <div class="collapsible">
                    <div class="collapsible-header" onclick="toggleCollapsible('advanced-classic-settings')">
                        <span class="arrow">▶</span> Advanced Settings
                    </div>
                    <div id="advanced-classic-settings" class="collapsible-content" style="display: none;">
                        <div class="row">
                          <div>
                            <label>Dark Alpha</label>
                            <input name="dark_alpha" type="number" min="0" max="1" step="0.01" value="{{dark_alpha}}" />
                          </div>
                          <div>
                            <label>Light Alpha</label>
                            <input name="light_alpha" type="number" min="0" max="1" step="0.01" value="{{light_alpha}}" />
                          </div>
                        </div>

                        <div class="row3">
                          <div>
                            <label>Rounded</label>
                            <input name="rounded" type="number" min="0" max="160" step="1" value="{{rounded}}" />
                          </div>
                          <div>
                            <label>Strength</label>
                            <input name="strength" type="number" min="0" max="2" step="0.1" value="{{strength}}" />
                          </div>
                          <div>
                            <label>Texture</label>
                            <input name="texture" type="number" min="0" max="1" step="0.05" value="{{texture}}" />
                          </div>
                        </div>

                        <label style="margin-top: 12px;">
                          <input name="preserve_finders" type="checkbox" value="1" {% if preserve_finders %}checked{% endif %} style="width: auto; margin-right: 8px;" />
                          Preserve finder patterns for better scanning
                        </label>
                    </div>
                </div>
            </div>

            <button class="btn" type="submit"> Generate QR Art</button>
            <div class="hint" style="text-align: center;">
              ⏱️ This may take 30-60 seconds. Please be patient!
            </div>
            {% if error %}
              <div class="error">{{error}}</div>
            {% endif %}
          </form>
        </div>
        
        <div class="card preview">
          <div class="imgbox">
            {% if img_data %}
              <img alt="QR Art Result" src="data:image/png;base64,{{img_data}}" />
            {% else %}
              <div class="foot" style="padding: 40px 20px;">
                <div style="font-size: 48px; margin-bottom: 16px;">🖼️</div>
                <div style="font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 8px;">Your QR Art will appear here</div>
                <div style="font-size: 12px; color: var(--muted);">Click "Generate QR Art" to create your masterpiece</div>
              </div>
            {% endif %}
          </div>
          {% if img_data %}
            <a class="dl" download="qr-art.png" href="data:image/png;base64,{{img_data}}">⬇️ Download PNG</a>
          {% endif %}
        </div>
      </div>
    </div>
    <script>setMode('ai');</script>
  </body>
</html>
"""

def _get_form_values() -> FormValues:
    data = (request.form.get("data") or "").strip()
    size = int(request.form.get("size") or 1024)
    border = int(request.form.get("border") or 4)
    dark = (request.form.get("dark") or "#000000").strip()
    light = (request.form.get("light") or "#ffffff").strip()
    dark_alpha = float(request.form.get("dark_alpha") or 0.42)
    light_alpha = float(request.form.get("light_alpha") or 0.06)
    rounded = int(request.form.get("rounded") or 44)
    strength = float(request.form.get("strength") or 1.0)
    texture = float(request.form.get("texture") or 0.85)
    
    # Check mode
    mode = request.form.get("mode") or "organic"
    if mode not in ['ai', 'organic', 'sharp']:
         # Fallback for classic dropdown
         mode = request.form.get("classic_mode") or "organic"

    preserve_finders = bool(request.form.get("preserve_finders"))
    
    # AI Params
    prompt = request.form.get("prompt")
    negative_prompt = request.form.get("negative_prompt")
    cn_scale = float(request.form.get("cn_scale") or 1.35)
    
    guidance_scale = float(request.form.get("guidance_scale") or 7.5)
    control_end = float(request.form.get("control_end") or 1.0)
    
    # Hack: return extended values object or just access request directly in generate
    # For now we attach them to the existing object structure by monkeypatching or just relying on request in generate()
    # Let's keep the FormValues clean and handle AI params separately in generate()
    
    return FormValues(
        data=data,
        size=size,
        border=border,
        dark=dark,
        light=light,
        dark_alpha=dark_alpha,
        light_alpha=light_alpha,
        rounded=rounded,
        preserve_finders=preserve_finders,
        strength=strength,
        texture=texture,
        mode=mode,
        readability=request.form.get("readability", "balanced"),
        guidance_scale=guidance_scale,
        control_end=control_end,
    )

def _render_page(*, values: FormValues, img_data: str | None, error: str | None) -> str:
    from flask import render_template_string
    
    # Get AI params from request if available to repopulate form
    prompt = request.form.get("prompt", "")
    negative_prompt = request.form.get("negative_prompt", "ugly, blurry, low quality")
    cn_scale = request.form.get("cn_scale", 1.35)
    performance_mode = request.form.get("performance_mode", "balanced")

    return render_template_string(
        _PAGE,
        data=values.data,
        size=values.size,
        border=values.border,
        dark=values.dark,
        light=values.light,
        dark_alpha=f"{values.dark_alpha:.2f}",
        light_alpha=f"{values.light_alpha:.2f}",
        rounded=values.rounded,
        preserve_finders=values.preserve_finders,
        strength=f"{values.strength:.2f}",
        texture=f"{values.texture:.2f}",
        mode=values.mode,
        readability=values.readability,
        prompt=prompt,
        negative_prompt=negative_prompt,
        cn_scale=cn_scale,
        performance_mode=performance_mode,
        img_data=img_data,
        error=error,
    )


def run_ai_task(task_id, values, prompt, negative_prompt, cn_scale, seed_val, performance_mode, guidance_scale, control_end):
    try:
        import qrcode
        from ai_generator import ai_engine
        
        print(f"Starting Task {task_id}")

        # 1. Generate Control Image
        # CRITICAL FIX: Border=0 for Full Bleed Natural QR
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=0,  # ZER0 BORDER!
        )
        qr.add_data(values.data)
        qr.make(fit=True)
        
        control_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        # 2. Determine parameters based on readability mode (SAME AS SYNC ENDPOINT)
        # Use the value sent by frontend (which is now auto-updated by JS)
        cn_scale_final = cn_scale
        control_end_final = 1.0
        
        # Context for Post-Processing
        smart_contrast = 1.0
        smart_sharpness = 1.0

        if values.readability == 'hidden':
            # HIDDEN MODE STRATEGY: SMART ADAPTIVE SHADOW ART
            
            # Analyze Prompt
            smart_settings = smart_analyze_prompt(prompt)
            print(f"[{task_id}] Smart Analysis: {smart_settings['mode']} (CN: {smart_settings['cn_scale']})")
            
            # Logic: If user kept default (1.35), use SMART value. If user changed it, trust user.
            if cn_scale == 1.35:
                 cn_scale_final = smart_settings['cn_scale']
            else:
                 cn_scale_final = cn_scale # User manual override
            
            # Save for later
            smart_contrast = smart_settings['contrast']
            smart_sharpness = smart_settings['sharpness']

            # Use SMART Blend
            blend_opacity = smart_settings.get('blend', 0.0)
            steps = 40
            
            # INJECT LIGHTING PROMPTS TO FORCE SCANNABILITY
            lighting_boost = "high contrast, deep shadows, volumetric lighting, sunlit, distinct light and dark areas, chiaroscuro"
            prompt = f"{prompt}, {lighting_boost}"

            anti_obvious = [
                "obvious grid", "regular squares", "artificial pattern", 
                "computer generated grid", "barcode appearance", "border", "frame", "flat lighting", "low contrast"
            ]
            negative_prompt = f"{negative_prompt}, {', '.join(anti_obvious)}"
            
        elif values.readability == 'scannable':
            blend_opacity = 0.60
            steps = 35
            
        else:  # balanced
            blend_opacity = 0.20
            steps = 35

        # Callback for progress
        def progress_callback(step, timestep, latents):
            if TASKS.get(task_id, {}).get('status') == 'cancelled':
                raise InterruptedError("Generation Cancelled")
            
            current_step = step + 1
            total = TASKS[task_id]['total']
            percent = int((current_step / total) * 100)
            TASKS[task_id]['step'] = current_step
            TASKS[task_id]['progress'] = percent

        # 3. Set total steps
        TASKS[task_id]['total'] = steps
        
        # Magic Prompt Logic
        magic_suffixes = [
            "raw photo", "photorealistic", "8k uhd", "dslr", "soft lighting", 
            "high quality", "film grain", "Fujifilm XT3", "intricate details"
        ]
        to_append = [m for m in magic_suffixes if m.lower() not in prompt.lower()]
        if to_append:
            prompt = f"{prompt}, {', '.join(to_append)}"

        anti_cartoon = [
            "(deformed, distorted, disfigured:1.3)", "poorly drawn", "bad anatomy", "wrong anatomy", 
            "extra limb", "missing limb", "floating limbs", "(mutated hands and fingers:1.4)", 
            "disconnected limbs", "mutation", "mutated", "ugly", "disgusting", "blurry", "amputation", 
            "(cartoon, anime, 3d, painting, drawing, illustration, sketch, flat, vector art:1.2)",
            "bad quality", "low quality", "jpeg artifacts"
        ]
        if negative_prompt:
             negative_prompt = f"{negative_prompt}, {', '.join(anti_cartoon)}"
        else:
             negative_prompt = ", ".join(anti_cartoon)

        final_img = ai_engine.generate(
            control_image=control_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            controlnet_conditioning_scale=cn_scale_final,
            guidance_scale=guidance_scale,
            control_guidance_start=0.0,
            control_guidance_end=control_end_final,
            seed=seed_val,
            mode=performance_mode,
            num_inference_steps=steps,
            callback=progress_callback
        )

        if final_img is None:
            raise Exception("AI Generator returned no image")
        
        if TASKS.get(task_id, {}).get('status') == 'cancelled':
            return

        # 4. POST-PROCESSING: SMART ANDROID BOOSTER
        # Apply only if readability is 'hidden' for maximum effect
        if values.readability == 'hidden':
            # Boost Contrast (Smart Value)
            enhancer = ImageEnhance.Contrast(final_img)
            final_img = enhancer.enhance(smart_contrast)
            
            # Boost Sharpness (Smart Value)
            enhancer = ImageEnhance.Sharpness(final_img)
            final_img = enhancer.enhance(smart_sharpness)
        
        # Post-Processing with adaptive blending
        try:
            # Blend if needed (only if blend_opacity > 0)
            if blend_opacity > 0:
                final_img = blend_qr_contrast(final_img, control_image, values.data, opacity=blend_opacity)
        except Exception as pp_e:
            print(f"Post-processing warning: {pp_e}")
            # If blending fails, use the image as is
            final_img = final_img

        # 5. Save & Finish
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        
        TASKS[task_id]['result'] = img_b64
        TASKS[task_id]['status'] = 'completed'
        TASKS[task_id]['progress'] = 100
        
    except InterruptedError:
        print(f"Task {task_id} Cancelled")
        TASKS[task_id]['status'] = 'cancelled'
    except Exception as e:
        print(f"Task {task_id} Failed: {e}")
        import traceback
        traceback.print_exc()
        TASKS[task_id]['status'] = 'failed'
        TASKS[task_id]['error'] = str(e)

@app.route('/generate_ai', methods=['POST'])
def generate_ai_endpoint():
    values = _get_form_values()
    if not values.data:
        return jsonify({'error': 'Data required'}), 400
        
    prompt = request.form.get('prompt', '').strip()
    negative_prompt = request.form.get('negative_prompt', '').strip()
    cn_scale = float(request.form.get('cn_scale') or 1.35)
    seed_val = int(request.form.get('seed') or -1)
    performance_mode = request.form.get('performance_mode', 'balanced')
    guidance_scale = float(request.form.get('guidance_scale') or 7.5)
    control_end = float(request.form.get('control_end') or 1.0)
    
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        'status': 'pending',
        'progress': 0,
        'step': 0,
        'total': 30,
        'result': None,
        'error': None
    }
    
    thread = threading.Thread(target=run_ai_task, args=(task_id, values, prompt, negative_prompt, cn_scale, seed_val, performance_mode, guidance_scale, control_end))
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/progress/<task_id>')
def progress_endpoint(task_id):
    task = TASKS.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)

@app.route('/cancel/<task_id>', methods=['POST'])
def cancel_endpoint(task_id):
    if task_id in TASKS:
        TASKS[task_id]['status'] = 'cancelled'
    return jsonify({'status': 'cancelled'})


@app.get("/")
def index() -> str:
    values = FormValues(
        data="https://aserehe.com",
        size=1024,
        border=4,
        dark="#000000",
        light="#ffffff",
        dark_alpha=0.50,
        light_alpha=0.10,
        rounded=44,
        preserve_finders=True,
        strength=1.1,
        texture=1.0,
        mode="organic",
        readability="balanced",
        guidance_scale=7.5,
        control_end=1.0,
    )

    return _render_page(values=values, img_data=None, error=None)


@app.post("/generate")
def generate() -> str:
    # --- CACHE CHECK ---
    # Create a unique signature based on all form inputs
    # We exclude file uploads from hash for simplicity (or read them if needed, but classic mode is fast anyway)
    # Focus on AI mode which is slow.
    
    # Sort keys to ensure consistent order
    form_data_str = "|".join([f"{k}:{v}" for k, v in sorted(request.form.items())])
    cache_key = hashlib.md5(form_data_str.encode()).hexdigest()
    
    if cache_key in RESULT_CACHE:
        print(f"Cache HIT for key: {cache_key}")
        # Return cached result immediately
        values = _get_form_values()
        return _render_page(values=values, img_data=RESULT_CACHE[cache_key], error=None)
    
    print(f"Cache MISS for key: {cache_key}")

    values = _get_form_values()
    if not values.data:
        return _render_page(values=values, img_data=None, error="Data tidak boleh kosong")

    # --- AI ControlNet Mode ---
    if values.mode == 'ai':
        # Guidance Scale & Control End from form
        guidance_scale = float(request.form.get("guidance_scale") or 7.5)
        control_end = float(request.form.get("control_end") or 1.0)
        
        prompt = request.form.get('prompt', '').strip()
        negative_prompt = request.form.get('negative_prompt', '').strip()
        cn_scale = float(request.form.get('cn_scale') or 1.35)
        seed_val = int(request.form.get('seed') or -1)
        performance_mode = request.form.get('performance_mode', 'balanced')

        if not prompt:
            return _render_page(values=values, img_data=None, error="Prompt is required for AI mode")

        # --- MAGIC PROMPT FOR REALISM ---
        # Force realistic style unless user explicitly overrides
        magic_suffixes = [
            "raw photo", "photorealistic", "8k uhd", "dslr", "soft lighting", 
            "high quality", "film grain", "Fujifilm XT3", "intricate details"
        ]
        
        # Only append if not already present
        to_append = [m for m in magic_suffixes if m.lower() not in prompt.lower()]
        if to_append:
            prompt = f"{prompt}, {', '.join(to_append)}"

        # --- MAGIC NEGATIVE PROMPT ---
        # Strong anti-cartoon protection
        anti_cartoon = [
            "(deformed, distorted, disfigured:1.3)", "poorly drawn", "bad anatomy", "wrong anatomy", 
            "extra limb", "missing limb", "floating limbs", "(mutated hands and fingers:1.4)", 
            "disconnected limbs", "mutation", "mutated", "ugly", "disgusting", "blurry", "amputation", 
            "(cartoon, anime, 3d, painting, drawing, illustration, sketch, flat, vector art:1.2)",
            "bad quality", "low quality", "jpeg artifacts"
        ]
        
        # Merge with user negative prompt
        if negative_prompt:
             negative_prompt = f"{negative_prompt}, {', '.join(anti_cartoon)}"
        else:
             negative_prompt = ", ".join(anti_cartoon)

        try:
            import qrcode
            from ai_generator import ai_engine

            # 1. Generate Control Image
            # CRITICAL FIX: Border=0 to prevent "white box" overlay effect!
            # We want the QR pattern to fill the canvas naturally (Full Bleed)
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=0,  # ZER0 BORDER = NO BOX ARTIFACTS!
            )
            qr.add_data(values.data)
            qr.make(fit=True)
            
            control_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")

            # 2. Determine parameters based on readability mode
            # NOW SMART: We analyze the prompt to adapt strictness!
            
            # Default values (overridden below)
            cn_scale_final = cn_scale
            control_end_final = 1.0
            
            # Context for Post-Processing
            smart_contrast = 1.0
            smart_sharpness = 1.0
            
            if values.readability == 'hidden':
                # HIDDEN MODE STRATEGY: SMART ADAPTIVE SHADOW ART
                
                # Analyze Prompt
                smart_settings = smart_analyze_prompt(prompt)
                print(f"[{threading.current_thread().name}] Smart Analysis: {smart_settings['mode']} (CN: {smart_settings['cn_scale']})")
                
                # Logic: If user kept default (1.35), use SMART value. If user changed it, trust user.
                if cn_scale == 1.35:
                     cn_scale_final = smart_settings['cn_scale']
                else:
                     cn_scale_final = cn_scale # User manual override
                
                # Post-Processing Values from Smart Analysis
                smart_contrast = smart_settings['contrast']
                smart_sharpness = smart_settings['sharpness']

                control_end_final = 1.0
                # Use SMART Blend (No longer hardcoded 0.0)
                # This allows micro-blending (0.15) for smooth subjects
                blend_opacity = smart_settings.get('blend', 0.0)
                steps = 40
                
                # INJECT LIGHTING & TEXTURE PROMPTS
                # If it's a "Nature" mode from Smart Analysis, force LUSH visuals
                if smart_settings['mode'] == "Textured/Nature":
                     nature_boost = "lush green foliage, dense ferns, mossy texture, biology, detailed leaves, sun rays in forest, organic pattern"
                     prompt = f"{prompt}, {nature_boost}"
                
                # General Lighting Boost for Scannability
                lighting_boost = "high contrast, deep shadows, volumetric lighting, sunlit, distinct light and dark areas, chiaroscuro"
                prompt = f"{prompt}, {lighting_boost}"

                anti_obvious = [
                    "obvious grid", "regular squares", "artificial pattern", 
                    "computer generated grid", "barcode appearance", "border", "frame", "flat lighting", "low contrast"
                ]
                negative_prompt = f"{negative_prompt}, {', '.join(anti_obvious)}"
                
            elif values.readability == 'scannable':
                # SCANNABLE: Standard strong mode with helper blending
                blend_opacity = 0.60
                steps = 35
                
            else:  # balanced (default)
                blend_opacity = 0.20
                steps = 35

            # 3. Run AI Generation
            final_img = ai_engine.generate(
                control_image=control_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=cn_scale_final,
                guidance_scale=guidance_scale,
                control_guidance_start=0.0,
                control_guidance_end=control_end_final,
                num_inference_steps=steps,
                seed=seed_val,
                mode=performance_mode
            )

            if final_img is None:
                raise Exception("AI Generator returned no image. Check terminal for memory errors.")
            
            # 4. POST-PROCESSING: SMART ANDROID BOOSTER
            # We trust the Smart Analyzer to give us the right values
            if values.readability == 'hidden':
                 # Boost Contrast
                 enhancer = ImageEnhance.Contrast(final_img)
                 final_img = enhancer.enhance(smart_contrast)
                 
                 # Boost Sharpness
                 enhancer = ImageEnhance.Sharpness(final_img)
                 final_img = enhancer.enhance(smart_sharpness)
            
            # 5. Blend if needed (Only for Scannable/Balanced modes, Hidden uses 0.0)
            if blend_opacity > 0:
                 try:
                     final_img = blend_qr_contrast(final_img, control_image, values.data, opacity=blend_opacity)
                 except Exception as pp_e:
                     print(f"Post-processing warning: {pp_e}")

            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            
            # Store in cache
            RESULT_CACHE[cache_key] = img_b64
            
            return _render_page(values=values, img_data=img_b64, error=None)

        except Exception as e:
            import traceback
            traceback.print_exc() # Print full error to terminal
            return _render_page(values=values, img_data=None, error=f"AI Error: {str(e)}")

    # --- Standard Frequency Separation Mode ---
    bg = None
    file = request.files.get("image")
    if file and file.filename:
        raw = file.read()
        try:
            bg = Image.open(io.BytesIO(raw))
            bg.load()
        except Exception:
            return _render_page(values=values, img_data=None, error="File gambar tidak valid")

    try:
        style = Style(
            dark_alpha=_clamp01(values.dark_alpha),
            light_alpha=_clamp01(values.light_alpha),
            rounded_radius=max(0, values.rounded),
            preserve_finders=values.preserve_finders,
            strength=_clamp01(values.strength),
            texture=_clamp01(values.texture),
            mode=values.mode,
        )

        img = generate_art_qr(
            data=values.data,
            background=bg,
            out_size=max(256, min(2048, values.size)),
            border_modules=max(0, min(10, values.border)),
            dark_color=_parse_color(values.dark),
            light_color=_parse_color(values.light),
            style=style,
        )
    except Exception as exc:
        return _render_page(values=values, img_data=None, error=str(exc))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    
    # Cache organic result too
    RESULT_CACHE[cache_key] = img_b64
    
    return _render_page(values=values, img_data=img_b64, error=None)


def main() -> int:
    # Use PORT 5000 (Standard Flask) to avoid 8080 conflicts in Colab
    app.run(host="0.0.0.0", port=5000, debug=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
