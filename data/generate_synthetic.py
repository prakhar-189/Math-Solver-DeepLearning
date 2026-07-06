# data/generate_synthetic.py
# -------------------------------------------------
# Generates a synthetic handwritten-style dataset of FLAT arithmetic
# expressions. This replaces the abandoned generator the README originally
# described. Flat left-to-right arithmetic is exactly what a CRNN+CTC model
# can represent (unlike the 2D-structured CROHME notation that broke the
# earlier version of this project), and generating it synthetically gives
# unlimited, balanced, non-repetitive data -- directly addressing the
# data-scarcity and overfitting problems that sank the CROHME attempt.
#
# Each expression is rendered character-by-character using a random handwriting
# font, with per-glyph size/rotation/baseline jitter and image-level
# augmentation, so the result looks like handwriting rather than clean typeset
# text -- narrowing the synthetic->real gap measured against the CROHME test set.
#
# Multiplication/division are RENDERED as the handwritten glyphs "×"/"÷" but
# LABELED as "*"/"/" so the label stays directly sympy-evaluable.
# -------------------------------------------------

import argparse
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = r"C:\Windows\Fonts"
HANDWRITING_FONTS = [
    "comic.ttf", "comicbd.ttf", "comici.ttf",
    "segoesc.ttf", "segoescb.ttf",
    "segoepr.ttf", "segoeprb.ttf",
    "Inkfree.ttf", "ITCKRIST.TTF", "LHANDW.TTF",
]

DISPLAY_MAP = {"*": "×", "/": "÷"}  # label char -> rendered glyph


def _available_fonts():
    fonts = []
    for name in HANDWRITING_FONTS:
        path = os.path.join(FONT_DIR, name)
        if os.path.exists(path):
            fonts.append(path)
    if not fonts:
        raise RuntimeError("No handwriting fonts found in C:\\Windows\\Fonts")
    return fonts


def random_number(rng: random.Random) -> str:
    if rng.random() < 0.12:
        a = rng.randint(0, 99)
        b = rng.randint(1, 99)
        return f"{a}.{b}"
    ndigits = rng.choice([1, 1, 2, 2, 3])
    return str(rng.randint(0, 10 ** ndigits - 1))


def random_label(rng: random.Random) -> str:
    """Return a flat arithmetic label string (using * and /)."""
    style = rng.random()

    # ~40%: integer equation with an equals and computed result (mimics CROHME "2+2=4")
    if style < 0.40:
        a = rng.randint(0, 50)
        b = rng.randint(0, 50)
        op = rng.choice(["+", "-", "*"])
        result = {"+": a + b, "-": a - b, "*": a * b}[op]
        return f"{a}{op}{b}={result}"

    # ~60%: expression to be solved (may include parens, division, decimals)
    n_ops = rng.choice([1, 1, 2, 2, 3])
    parts = [random_number(rng)]
    for _ in range(n_ops):
        parts.append(rng.choice(["+", "-", "*", "/"]))
        parts.append(random_number(rng))
    expr = "".join(parts)

    # ~35% of expressions: wrap a leading pair in parentheses, e.g. (4+3)*8
    if n_ops >= 2 and rng.random() < 0.35:
        # find index of second operator to close the paren after first operand-op-operand
        # parts = [num, op, num, op, num, ...]; wrap parts[0:3]
        head = "(" + "".join(parts[0:3]) + ")"
        expr = head + "".join(parts[3:])
    return expr


def render_expression(label: str, fonts, rng: random.Random, height: int = 64) -> np.ndarray:
    """Render a label to a black-on-white grayscale image with handwriting jitter."""
    font_path = rng.choice(fonts)
    base_size = rng.randint(38, 52)

    glyph_imgs = []
    for ch in label:
        disp = DISPLAY_MAP.get(ch, ch)
        size = int(base_size * rng.uniform(0.85, 1.15))
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox(disp)
        w = max(bbox[2] - bbox[0], 1)
        h = max(bbox[3] - bbox[1], 1)
        pad = int(size * 0.3)
        canvas = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
        draw = ImageDraw.Draw(canvas)
        draw.text((pad - bbox[0], pad - bbox[1]), disp, fill=0, font=font)
        # per-glyph rotation jitter
        angle = rng.uniform(-9, 9)
        canvas = canvas.rotate(angle, expand=True, fillcolor=255)
        glyph_imgs.append(np.array(canvas))

    # compose horizontally with jittered spacing and vertical baseline offset
    max_h = max(g.shape[0] for g in glyph_imgs)
    total_w = sum(g.shape[1] for g in glyph_imgs) + int(len(glyph_imgs) * base_size * 0.12)
    canvas_h = max_h + int(base_size * 0.5)
    canvas = np.full((canvas_h, total_w + base_size), 255, dtype=np.uint8)

    x = rng.randint(2, base_size // 2)
    for g in glyph_imgs:
        gh, gw = g.shape
        y = (canvas_h - gh) // 2 + rng.randint(-base_size // 6, base_size // 6)
        y = max(0, min(canvas_h - gh, y))
        # darken-min blend so overlapping strokes stay dark
        region = canvas[y:y + gh, x:x + gw]
        canvas[y:y + gh, x:x + gw] = np.minimum(region, g[:, :region.shape[1]])
        x += gw + rng.randint(-int(base_size * 0.05), int(base_size * 0.14))
        if x >= canvas.shape[1] - base_size:
            break

    return canvas


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Image-level augmentation: stroke thickness, slight rotation, blur, noise."""
    # stroke thickness (dilate = thicker black strokes on white, erode = thinner)
    k = rng.choice([1, 1, 2])
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        if rng.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)   # thinner strokes
        else:
            img = cv2.dilate(img, kernel, iterations=1)   # thicker strokes

    # small global rotation
    ang = rng.uniform(-4, 4)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # occasional blur
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # gaussian noise
    if rng.random() < 0.5:
        noise = rng.uniform(3, 12)
        img = np.clip(img.astype(np.float32) + np.random.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000, help="number of training images")
    parser.add_argument("--n-val", type=int, default=2000, help="number of validation images")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fonts = _available_fonts()
    print(f"Using {len(fonts)} handwriting fonts.")

    for split, count, seed in [("train", args.n, args.seed), ("val", args.n_val, args.seed + 1)]:
        rng = random.Random(seed)
        np.random.seed(seed)
        split_dir = os.path.join(args.out_dir, split)
        img_dir = os.path.join(split_dir, "img")
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(split_dir, "labels.txt"), "w", encoding="utf-8") as lf:
            for i in range(count):
                label = random_label(rng)
                img = render_expression(label, fonts, rng)
                img = augment(img, rng)
                name = f"{i:06d}.png"
                cv2.imwrite(os.path.join(img_dir, name), img)
                lf.write(f"{name}\t{label}\n")
                if (i + 1) % 2000 == 0:
                    print(f"  [{split}] {i + 1}/{count}")
        print(f"Wrote {count} {split} images to {split_dir}")


if __name__ == "__main__":
    main()
