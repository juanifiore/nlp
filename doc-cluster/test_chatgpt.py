import pandas as pd
import numpy as np

df = pd.read_csv('datasets/topic_sentences_labels.csv')

docs = df['text'].tolist()
labels =  df['label']
labels_str = df['label_str']



