import pandas as pd
from transformers import MarianTokenizer

# Cargar el tokenizador de MarianMT (Inglés a Español)
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Obtener el vocabulario completo (es un diccionario token: índice)
vocab = tokenizer.get_vocab()

# Convertir el vocabulario en una lista de tuplas para crear un DataFrame
vocab_items = [(token, index) for token, index in vocab.items()]

# Crear un DataFrame con el vocabulario
df_vocab = pd.DataFrame(vocab_items, columns=['Token', 'Index'])

# Guardar el DataFrame en un archivo CSV
df_vocab.to_csv('marianmt_vocab.csv', index=False)

print("Vocabulario guardado en 'marianmt_vocab.csv'")

