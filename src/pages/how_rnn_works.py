from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="How RNNs Work", layout="centered")
    st.title("How RNNs Work (Mini Blog)")

    st.markdown(
        """
A **Recurrent Neural Network (RNN)** is a neural network designed for **sequences** (text, audio, time series).
Unlike a standard feed-forward network, an RNN keeps a *memory* of what it has seen so far.




## How text generation works here
This project is a **character-level language model**:

1. Take a short seed text window (here it’s 40 characters).
2. Predict a probability distribution for the next character.
3. Sample one character from that distribution.
4. Append it, slide the window forward by one, and repeat.

## What “temperature” means
The model outputs probabilities, but **temperature** changes how “random” sampling feels:

- Lower temperature (e.g. 0.2–0.5): safer, more repetitive
- Higher temperature (e.g. 0.8–1.2): more surprising, more chaotic

It does this by reshaping the probability distribution before sampling.

---

Use the main page to generate text with your chosen temperature and output length.
"""
    )

    if st.button("Back to generator"):
        st.switch_page("main.py")


if __name__ == "__main__":
    main()
