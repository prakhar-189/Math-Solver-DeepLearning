# inference/latex_converter.py
# ------------------------------------------
# Converts a recognized ASCII expression into a LaTeX string for display.
from utils import normalize_equation


def to_latex(eq: str) -> str:
    if not eq:
        return ""
    eq = normalize_equation(eq).replace(" ", "")
    # Prettier operators for rendering (the ASCII form is kept for evaluation).
    eq = eq.replace("*", r" \times ").replace("/", r" \div ")
    return eq
