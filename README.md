<!-- @format -->

# RNN-Poetic-Text

## Streamlit app

View the app at https://rnn-poetic-text-a9r4ctoxqtp4xjteahasop.streamlit.app/ or run it locally.

Run the UI:

1. Install Streamlit (if needed):
   - `pip install streamlit`
2. Start the app:
   - `streamlit run src/main.py`

The app has two inputs:

- Temperature
- Length output

There is also a "How RNN Works" page linked from the main screen.

The trained Keras model is stored at `models/shakespeare_model.keras`.

## Streamlit Cloud

If deploying to Streamlit Cloud, dependencies are installed from `requirements.txt` and Python is pinned via `runtime.txt`.
