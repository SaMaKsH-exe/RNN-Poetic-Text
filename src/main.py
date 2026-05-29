from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import RMSprop


SEQUENCE_LENGTH = 40
STEP_SIZE = 3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "shakespeare_model.keras"

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.01"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "5"))


@st.cache_data(show_spinner=False)
def load_corpus() -> tuple[str, list[str], dict[str, int], dict[int, str]]:
    filepath = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
    )
    text = open(filepath, "rb").read().decode(encoding="utf-8").lower()
    text = text[300000:800000]

    characters = sorted(set(text))
    char_to_index = {ch: i for i, ch in enumerate(characters)}
    index_to_char = {i: ch for ch, i in char_to_index.items()}
    return text, characters, char_to_index, index_to_char


def vectorize(
    text: str,
    characters: list[str],
    char_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    sentences: list[str] = []
    next_chars: list[str] = []

    for i in range(0, len(text) - SEQUENCE_LENGTH, STEP_SIZE):
        sentences.append(text[i: i + SEQUENCE_LENGTH])
        next_chars.append(text[i + SEQUENCE_LENGTH])

    x = np.zeros((len(sentences), SEQUENCE_LENGTH,
                 len(characters)), dtype=bool)
    y = np.zeros((len(sentences), len(characters)), dtype=bool)

    for i, sentence in enumerate(sentences):
        for t, ch in enumerate(sentence):
            x[i, t, char_to_index[ch]] = True
        y[i, char_to_index[next_chars[i]]] = True

    return x, y


def build_model(vocab_size: int) -> tf.keras.Model:
    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, vocab_size)))
    model.add(LSTM(128))
    model.add(Dense(vocab_size, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(learning_rate=LEARNING_RATE),
    )
    return model


@st.cache_resource(show_spinner=False)
def load_model_cached(model_mtime: float) -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH)


def ensure_model(
    text: str,
    characters: list[str],
    char_to_index: dict[str, int],
) -> tf.keras.Model:
    if not MODEL_PATH.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with st.spinner(
            f"Training model (first run, {TRAIN_EPOCHS} epochs)… This may take a while."
        ):
            x, y = vectorize(text, characters, char_to_index)
            model = build_model(vocab_size=len(characters))
            model.fit(x, y, batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS)
            model.save(MODEL_PATH)

    return load_model_cached(MODEL_PATH.stat().st_mtime)


def sample_index(preds: np.ndarray, temperature: float) -> int:
    temperature = float(temperature)
    if temperature <= 0:
        temperature = 1e-6

    preds = np.asarray(preds).astype(np.float64)
    preds = np.log(preds + 1e-8) / temperature
    preds = np.exp(preds)
    preds = preds / np.sum(preds)
    return int(np.random.choice(len(preds), p=preds))


def generate_text(
    model: tf.keras.Model,
    text: str,
    characters: list[str],
    char_to_index: dict[str, int],
    index_to_char: dict[int, str],
    length: int,
    temperature: float,
) -> tuple[str, str]:
    start_index = random.randint(0, len(text) - SEQUENCE_LENGTH - 1)
    seed = text[start_index: start_index + SEQUENCE_LENGTH]

    generated = seed
    sentence = seed

    for _ in range(length):
        x_pred = np.zeros((1, SEQUENCE_LENGTH, len(characters)), dtype=bool)
        for t, ch in enumerate(sentence):
            x_pred[0, t, char_to_index[ch]] = True

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, temperature=temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return seed, generated


def main() -> None:
    st.set_page_config(page_title="RNN Poetic Text", layout="centered")
    st.title("RNN Poetic Text Generator")

    if st.button("How RNN Works"):
        st.switch_page("pages/how_rnn_works.py")

    temperature_str = st.text_input("Temperature", value="0.5")
    length_str = st.text_input("Length output", value="400")

    try:
        temperature = float(temperature_str)
    except ValueError:
        st.error("Temperature must be a number (e.g., 0.5)")
        return

    try:
        length = int(length_str)
    except ValueError:
        st.error("Length output must be an integer (e.g., 400)")
        return

    if length < 1:
        st.error("Length output must be >= 1")
        return

    text, characters, char_to_index, index_to_char = load_corpus()
    model = ensure_model(text, characters, char_to_index)

    if st.button("Generate"):
        with st.spinner("Generating text…"):
            seed, generated = generate_text(
                model=model,
                text=text,
                characters=characters,
                char_to_index=char_to_index,
                index_to_char=index_to_char,
                length=length,
                temperature=temperature,
            )
        st.write("Seed:")
        st.code(seed)
        st.write("Output:")
        st.text_area("", value=generated, height=300)


if __name__ == "__main__":
    main()
