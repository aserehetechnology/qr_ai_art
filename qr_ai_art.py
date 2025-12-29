from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import qrcode
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageStat, ImageChops


@dataclass(frozen=True)
class Style:
    dark_alpha: float
    light_alpha: float
    rounded_radius: int
    preserve_finders: bool
    strength: float
    texture: float
    mode: str = "sharp"  # sharp, organic


def _create_soft_qr_mask(matrix: list[list[bool]], size: int, border: int) -> Image.Image:
    """Create a soft/blurred mask from the QR matrix to avoid grid artifacts."""
    modules = len(matrix)
    module_px = size // modules
    mask = Image.new("L", (size, size), 255)  # Default white (light)
    draw = ImageDraw.Draw(mask)

    for my in range(modules):
        for mx in range(modules):
            if matrix[my][mx]:  # Dark module
                x0 = mx * module_px
                y0 = my * module_px
                x1 = x0 + module_px
                y1 = y0 + module_px
                draw.rectangle((x0, y0, x1, y1), fill=0)  # Black for dark modules

    # Apply heavy blur to remove grid lines
    blur_radius = max(1, int(module_px * 0.85))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Enhance contrast to make transitions sharper but still curved
    mask = ImageEnhance.Contrast(mask).enhance(1.5)
    return mask



def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_color(color: str) -> tuple[int, int, int]:
    value = color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError("Color harus format #RRGGBB atau #RGB")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _make_placeholder_background(size: int) -> Image.Image:
    base = Image.new("RGB", (size, size), (20, 60, 25))
    draw = ImageDraw.Draw(base)
    for y in range(size):
        t = y / max(1, size - 1)
        r = int(20 + 30 * t)
        g = int(70 + 120 * t)
        b = int(25 + 35 * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b))
    base = base.filter(ImageFilter.GaussianBlur(radius=max(1, size // 320)))
    return base


def _qr_matrix(data: str, *, border: int) -> list[list[bool]]:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=1,
        border=0,
    )
    qr.add_data(data)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    n = len(matrix)
    full = [[False for _ in range(n + 2 * border)] for _ in range(n + 2 * border)]
    for y in range(n):
        for x in range(n):
            full[y + border][x + border] = bool(matrix[y][x])
    return full


def _is_in_square(x: int, y: int, left: int, top: int, size: int) -> bool:
    return left <= x < left + size and top <= y < top + size


def _is_finder_or_separator(x: int, y: int, *, n: int, border: int) -> bool:
    finder = 7
    sep = 1
    size = finder + 2 * sep
    tl_left = border - sep
    tl_top = border - sep
    tr_left = (n - border) - finder + 0 - sep
    tr_top = border - sep
    bl_left = border - sep
    bl_top = (n - border) - finder + 0 - sep
    return (
        _is_in_square(x, y, tl_left, tl_top, size)
        or _is_in_square(x, y, tr_left, tr_top, size)
        or _is_in_square(x, y, bl_left, bl_top, size)
    )


def _rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def _blend_patch(base: Image.Image, patch: Image.Image, alpha: float) -> Image.Image:
    if alpha <= 0:
        return base
    return Image.blend(base, patch, _clamp01(alpha))


def _module_mask(size: int, inset: int, radius: int) -> Image.Image:
    if inset <= 0 and radius <= 0:
        return Image.new("L", (size, size), 255)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    left = inset
    top = inset
    right = size - 1 - inset
    bottom = size - 1 - inset
    if right <= left or bottom <= top:
        return Image.new("L", (size, size), 255)
    if radius > 0:
        draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=255)
    else:
        draw.rectangle((left, top, right, bottom), fill=255)
    return mask


def _mean_luma(region: Image.Image) -> float:
    r, g, b = ImageStat.Stat(region).mean[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _towards_luma(region: Image.Image, target_luma: float, min_factor: float, max_factor: float) -> Image.Image:
    current = _mean_luma(region)
    if current <= 1:
        factor = 1.0
    else:
        factor = target_luma / current
    factor = max(min_factor, min(max_factor, factor))
    return ImageEnhance.Brightness(region).enhance(factor)

def _reduce_texture(region: Image.Image, texture: float) -> Image.Image:
    t = _clamp01(texture)
    if t >= 0.999:
        return region
    stat = ImageStat.Stat(region)
    mean = tuple(int(round(x)) for x in stat.mean[:3])
    flat = Image.new("RGB", region.size, mean)
    return Image.blend(flat, region, t)


def generate_art_qr(
    *,
    data: str,
    background: Image.Image | None,
    out_size: int,
    border_modules: int,
    dark_color: tuple[int, int, int],
    light_color: tuple[int, int, int],
    style: Style,
) -> Image.Image:
    matrix = _qr_matrix(data, border=border_modules)
    modules = len(matrix)

    module_px = out_size // modules
    if module_px <= 0:
        raise ValueError("out_size terlalu kecil untuk QR yang dihasilkan")
    size = module_px * modules

    if background is None:
        bg = _make_placeholder_background(size)
    else:
        bg = _center_crop_square(background.convert("RGB")).resize((size, size), Image.LANCZOS)

    bg = ImageEnhance.Color(bg).enhance(1.25)
    bg = ImageEnhance.Contrast(bg).enhance(1.08)
    
    # Pre-blur background as requested in specs ("Convert image -> grayscale + blur ringan")
    # We keep color but apply the blur to reduce noise before mapping
    bg = bg.filter(ImageFilter.GaussianBlur(radius=0.5))

    base_bg = bg.copy().convert("RGB")
    canvas = base_bg.copy()
    draw = ImageDraw.Draw(canvas)

    strength = _clamp01(style.strength)
    
    # --- PHASE 1/2 MVP LOGIC: Hard Constraint Mapping ---
    # Logic:
    # 1. Per module QR:
    #    - hitam -> ambil pixel gelap (force dark)
    #    - putih -> pixel terang (force light)
    # 2. Lock finder pattern
    
    for my in range(modules):
        for mx in range(modules):
            # Coordinates
            x0 = mx * module_px
            y0 = my * module_px
            x1 = x0 + module_px
            y1 = y0 + module_px
            
            # 1. Lock Finder Pattern & Separators (CRITICAL for scanability)
            if _is_finder_or_separator(mx, my, n=modules, border=border_modules):
                if style.preserve_finders:
                    is_dark = matrix[my][mx]
                    # Absolute Black/White for finders to guarantee detection
                    col = (0, 0, 0) if is_dark else (255, 255, 255)
                    draw.rectangle((x0, y0, x1, y1), fill=col)
                    continue

            # 2. Per Module Constraint
            is_dark = matrix[my][mx]
            region = base_bg.crop((x0, y0, x1, y1))
            
            if is_dark:
                # HITAM -> Ambil pixel gelap
                # Strategy: Darken the region significantly but keep some texture
                # Target brightness: very low
                
                # Formula: Region * 0.35 (adjustable by strength)
                # Base factor 0.35, modulated by strength
                factor = 0.35 + (1.0 - strength) * 0.3
                region = ImageEnhance.Brightness(region).enhance(factor)
                
                # Optional: Increase contrast to make features pop in the dark
                region = ImageEnhance.Contrast(region).enhance(1.1)
                
            else:
                # PUTIH -> Pixel terang
                # Strategy: Brighten the region
                # Target brightness: high
                
                # Formula: Region * 1.65 (adjustable by strength)
                factor = 1.65 + (strength - 0.5) * 0.5
                region = ImageEnhance.Brightness(region).enhance(factor)
                
                # Optional: Reduce saturation slightly in highlights to avoid weird artifacts
                region = ImageEnhance.Color(region).enhance(0.9)

            # Paste back
            canvas.paste(region, (x0, y0))

    if style.rounded_radius > 0:

        mask = _rounded_mask(size, style.rounded_radius)
        canvas = ImageOps.fit(canvas, (size, size), method=Image.LANCZOS)
        rgba = canvas.convert("RGBA")
        rgba.putalpha(mask)
        return rgba

    return canvas


def create_finder_mask(data: str, size: int, border: int) -> Image.Image:
    """Creates a transparent PNG with ONLY the finder patterns in black/white."""
    matrix = _qr_matrix(data, border=border)
    modules = len(matrix)
    module_px = size // modules
    real_size = module_px * modules
    
    # Create RGBA image, transparent background
    mask = Image.new("RGBA", (real_size, real_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)

    for my in range(modules):
        for mx in range(modules):
            # Check if finder
            if _is_finder_or_separator(mx, my, n=modules, border=border):
                x0 = mx * module_px
                y0 = my * module_px
                x1 = x0 + module_px
                y1 = y0 + module_px
                
                is_dark = matrix[my][mx]
                if is_dark:
                    draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 255))
                else:
                    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, 255))
    
    if real_size != size:
        mask = mask.resize((size, size), Image.NEAREST)
        
    return mask



def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("Harus bilangan bulat positif")
    return n


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qr_ai_art")
    parser.add_argument("--data", required=True, help="Teks/URL yang akan dimasukkan ke QR")
    parser.add_argument("--image", help="Path gambar latar (opsional)")
    parser.add_argument("--out", default="out.png", help="Path output (default: out.png)")
    parser.add_argument("--size", type=_positive_int, default=1024, help="Ukuran output px (default: 1024)")
    parser.add_argument("--border", type=int, default=4, help="Quiet zone (dalam modul) (default: 4)")
    parser.add_argument("--dark", default="#000000", help="Warna modul gelap (default: #000000)")
    parser.add_argument("--light", default="#ffffff", help="Warna modul terang (default: #ffffff)")
    parser.add_argument("--dark-alpha", type=float, default=0.62, help="Alpha modul gelap (0-1)")
    parser.add_argument("--light-alpha", type=float, default=0.18, help="Alpha modul terang (0-1)")
    parser.add_argument("--strength", type=float, default=1.0, help="Kekuatan pemaksaan (0-1)")
    parser.add_argument("--texture", type=float, default=0.85, help="Pertahankan tekstur (0-1)")
    parser.add_argument("--mode", default="sharp", choices=["sharp", "organic"], help="Mode blend: sharp atau organic")
    parser.add_argument("--round", type=int, default=36, help="Radius sudut (px) (default: 36)")

    parser.add_argument(
        "--no-preserve-finders",
        action="store_true",
        help="Jika diaktifkan, finder/timing ikut diblend (kurang aman untuk scan)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    bg = None
    if args.image:
        bg = Image.open(args.image)

    style = Style(
        dark_alpha=_clamp01(args.dark_alpha),
        light_alpha=_clamp01(args.light_alpha),
        rounded_radius=max(0, int(args.round)),
        preserve_finders=not bool(args.no_preserve_finders),
        strength=_clamp01(args.strength),
        texture=_clamp01(args.texture),
        mode=args.mode,
    )


    out_img = generate_art_qr(
        data=args.data,
        background=bg,
        out_size=args.size,
        border_modules=max(0, int(args.border)),
        dark_color=_parse_color(args.dark),
        light_color=_parse_color(args.light),
        style=style,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_img.mode == "RGBA" and out_path.suffix.lower() in {".jpg", ".jpeg"}:
        out_img = out_img.convert("RGB")
    out_img.save(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
