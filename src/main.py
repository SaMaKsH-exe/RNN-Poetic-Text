import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop


filepath = tf.keras.utils.get_file('shakespeare.txt',   
                                   'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt',)

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

print(f"Length of text: {len(text)} characters")
