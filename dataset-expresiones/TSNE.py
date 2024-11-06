import torch
from transformers import MarianMTModel, MarianTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


# Configuración de modelo y tokenizador
model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Función para obtener el embedding de una palabra
def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Embedding promedio

# Lista de palabras en inglés y español
words = [("water", "agua"), ("sun", "sol"), ("moon", "luna"), ("star", "estrella"),
         ("tree", "árbol"), ("flower", "flor"), ("house", "casa"), ("dog", "perro"),
         ("cat", "gato"), ("mountain", "montaña")]

# Extraer los embeddings
embeddings = []
labels = []

for eng_word, esp_word in words:
    embeddings.append(get_embedding(eng_word))
    labels.append(f"{eng_word} (EN)")
    embeddings.append(get_embedding(esp_word))
    labels.append(f"{esp_word} (ES)")


# Convertir la lista de embeddings a un array de NumPy
embeddings = np.array(embeddings)


# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=15)
embeddings_tsne = tsne.fit_transform(embeddings)

# Visualizar resultados
plt.figure(figsize=(10, 10))
for i, label in enumerate(labels):
    x, y = embeddings_tsne[i]
    plt.scatter(x, y)
    plt.text(x + 0.02, y + 0.02, label, fontsize=9)
plt.title("t-SNE Visualization of English-Spanish Word Embeddings")
plt.show()

