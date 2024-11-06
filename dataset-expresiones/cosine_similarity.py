import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer

# Cargar el modelo y el tokenizador
model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Función para obtener el embedding de una palabra
def get_word_embedding(word, context=None):
    # Si se proporciona un contexto, incluimos la palabra en una oración; de lo contrario, usamos solo la palabra.
    if context:
        sentence = context.replace("[WORD]", word)  # Reemplazamos un marcador por la palabra
    else:
        sentence = word  # Si no hay contexto, usamos solo la palabra

    # Tokenizar la frase o palabra
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # Obtener las salidas del encoder del modelo (última capa oculta)
    with torch.no_grad():
        outputs = model.model.encoder(**inputs)

    # Obtener los embeddings de la palabra solicitada
    # outputs.last_hidden_state tiene la forma (batch_size, sequence_length, hidden_size)
    embeddings = outputs.last_hidden_state.squeeze(0)

    # Encontramos el índice del token correspondiente a la palabra solicitada
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    word_embedding = None

    for idx, input_id in enumerate(inputs.input_ids[0]):
        if input_id in token_ids:
            word_embedding = embeddings[idx].numpy()
            break

    if word_embedding is not None:
        return word_embedding
    else:
        raise ValueError(f"El token '{word}' no fue encontrado en la secuencia tokenizada.")

# Ejemplo de uso:
#word = "cat"
#context = "The [WORD] is sleeping"  # Contexto opcional
#embedding = get_word_embedding(word, context)
#print(f"Embedding de '{word}':\n", embedding)


#======================================
# COSINE_SIMILARITY
#======================================

def cos_sim(word1, word2, context1=None, context2=None):

    if context2 == None:
        context2 = context1

    # con contexto
    if context1 != None:
        embedding_cat = get_word_embedding(word1, context1)
        embedding_dog = get_word_embedding(word2, context2)
        similarity = cosine_similarity(embedding_cat.reshape(1, -1), embedding_dog.reshape(1, -1))
        print(f"Cosine Similarity con contexto entre {word1} y {word2}: {similarity[0][0]}")
    else:
        similarity=None

    # embeddings fuera de contexto
    embedding_cat_ooc = get_word_embedding(word1)
    embedding_dog_ooc = get_word_embedding(word2)
    similarity_ooc = cosine_similarity(embedding_cat_ooc.reshape(1, -1), embedding_dog_ooc.reshape(1, -1))
    print(f"Cosine Similarity out of context entre {word1} y {word2}: {similarity_ooc[0][0]}")

    return similarity, similarity_ooc



