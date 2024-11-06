import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

# Cargar el modelo y el tokenizador de MarianMT para traduccion de ingles a español
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_sentence(sentence):
    # Tokenizar la entrada
    tokenized_input = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # Realizar la traduccion
    translated = model.generate(**tokenized_input)

    # Decodificar los tokens de salida
    translated_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_sentence

# Cargar el archivo CSV
csv_file = "ingles-espanol_jerga.csv"
#csv_out = "ingles-espanol-traduccion.csv"
df = pd.read_csv(csv_file)
df = df.head(1000)

# Asegurarse de que el archivo tiene una columna llamada "Ingles"
if 'Ingles' not in df.columns:
    raise ValueError("El archivo CSV debe tener una columna llamada 'Ingles'")

# Crear una nueva columna "Traduccion" para las traducciones
df['Traduccion argentina'] = df['Ingles'].apply(translate_sentence)

# Guardar el archivo con la nueva columna "Traduccion"
df.to_csv(csv_file, index=False)

print(f"Las traducciones han sido añadidas a '{csv_file}'.")

