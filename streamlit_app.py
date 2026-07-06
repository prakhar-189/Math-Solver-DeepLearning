# streamlit_app.py
# -----------------------------------------
# Streamlit UI: upload a handwritten arithmetic image -> recognize -> solve.
# All the image handling goes through the SAME utils.preprocess_image the model
# was trained/evaluated with, which is the fix for the original app returning
# garbage (it previously ran a different preprocessing path than training).
# -----------------------------------------

import numpy as np
import streamlit as st
from PIL import Image

from inference.latex_converter import to_latex
from inference.predict import predict_equation, solve_equation
from utils import preprocess_image

st.set_page_config(page_title="Handwritten Math Solver", page_icon="🧮")
st.title("🧮 Handwritten Math Solver")
st.write(
    """
    Upload an image of a **flat handwritten arithmetic expression** and the model will
    recognize and solve it.

    Supported: digits, `+ - × ÷ ( ) .` and `=`.  Examples: `12+5`, `(4+3)×8`, `5.25×3`.

    *Scope note: this recognizes flat left-to-right arithmetic only — not fractions,
    exponents, or algebra. See the README for why.*
    """
)

uploaded = st.file_uploader("Upload handwritten equation", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    display = img.copy()
    display.thumbnail((600, 600))
    st.image(display, caption="Uploaded image")

    with st.spinner("Recognizing..."):
        try:
            # Show exactly what the model receives (canonical preprocessing).
            processed = preprocess_image(img_np)
            st.image(processed[:, :, 0], caption="Model input (after preprocessing)", clamp=True, width=400)

            equation = predict_equation(img_np)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:  # noqa: BLE001
            st.error(f"Recognition failed: {e}")
            st.stop()

    if not equation:
        st.warning("No expression detected. Try clearer, well-spaced handwriting.")
        st.stop()

    st.subheader("Recognized expression")
    st.code(equation, language="text")

    st.subheader("LaTeX")
    st.latex(to_latex(equation))

    st.subheader("Result")
    result, err = solve_equation(equation)
    if err:
        st.warning(err)
    else:
        st.success(result)
