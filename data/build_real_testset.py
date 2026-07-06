# data/build_real_testset.py
# -------------------------------------------------
# Extracts the flat-arithmetic subset of the CROHME handwritten-math dataset
# into a small held-out REAL test set. CROHME is ~70% 2D-structured notation
# (fractions, exponents, subscripts) that a CTC recognizer cannot represent;
# only the flat left-to-right arithmetic expressions are in-scope for this
# project, so this script filters to those and normalizes their LaTeX-token
# labels into the compact character vocabulary the model actually predicts.
#
# Source (not committed): E:\...\Math-Solver-DeepLearning\Dataset\CROHME
# Output: data/real_test/{img/, labels.txt}
# -------------------------------------------------

import argparse
import os

# LaTeX-token -> canonical character. Handwritten "×"/"÷" map to "*"/"/" so the
# model shares one class per operator regardless of how it was written, and the
# result stays directly sympy-evaluable.
TOKEN_MAP = {
    "\\times": "*",
    "\\div": "/",
    "\\cdot": "*",
}
# Canonical vocabulary (what a "flat arithmetic" expression may contain).
ALLOWED_CHARS = set("0123456789+-*/().=")


def normalize_label(latex_label: str):
    """Convert a space-separated CROHME LaTeX label into a compact string.
    Returns None if the expression is not flat arithmetic (contains any token
    outside the allowed arithmetic vocabulary, e.g. a variable, \\frac, ^, _)."""
    out = []
    for tok in latex_label.split():
        tok = TOKEN_MAP.get(tok, tok)
        if len(tok) == 1 and tok in ALLOWED_CHARS:
            out.append(tok)
        else:
            return None  # non-arithmetic token -> reject whole expression
    s = "".join(out)
    # Require at least one digit and one operator so we keep real expressions,
    # not stray single symbols.
    if not any(c.isdigit() for c in s):
        return None
    if not any(c in "+-*/" for c in s):
        return None
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crohme-dir",
        default=r"E:\Data Science Materials\Additional Projects\Math-Solver-DeepLearning\Dataset\CROHME",
    )
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "real_test"))
    parser.add_argument("--splits", nargs="+", default=["train", "2014", "2016", "2019"])
    args = parser.parse_args()

    out_img = os.path.join(args.out_dir, "img")
    os.makedirs(out_img, exist_ok=True)

    entries = []
    seen_labels = set()
    for split in args.splits:
        cap = os.path.join(args.crohme_dir, split, "caption.txt")
        img_dir = os.path.join(args.crohme_dir, split, "img")
        if not os.path.exists(cap):
            print(f"[skip] {cap} not found")
            continue
        with open(cap, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    parts = line.rstrip("\n").split(maxsplit=1)
                if len(parts) < 2:
                    continue
                cid, latex = parts[0], parts[1]
                label = normalize_label(latex)
                if label is None:
                    continue
                stem = os.path.splitext(os.path.basename(cid))[0]
                src_img = os.path.join(img_dir, stem + ".bmp")
                if not os.path.exists(src_img):
                    continue
                dst_name = f"{split}__{stem}.png"
                entries.append((dst_name, label, src_img))
                seen_labels.add(label)

    # Copy images (converting .bmp -> .png so the whole pipeline is one format)
    import cv2

    written = 0
    with open(os.path.join(args.out_dir, "labels.txt"), "w", encoding="utf-8") as lf:
        for dst_name, label, src_img in entries:
            img = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            cv2.imwrite(os.path.join(out_img, dst_name), img)
            lf.write(f"{dst_name}\t{label}\n")
            written += 1

    print(f"Wrote {written} real flat-arithmetic test images ({len(seen_labels)} unique expressions) to {args.out_dir}")


if __name__ == "__main__":
    main()
