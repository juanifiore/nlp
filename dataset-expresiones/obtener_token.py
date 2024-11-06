from transformers import MarianTokenizer

# Cargar el tokenizador de MarianMT (ejemplo: de inglés a español)
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)

def tokenize_sentence(sentence):
    # Tokenizar la frase y devolver los tokens y sus IDs
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids

# Input: Frase que deseas tokenizar
input_sentence = "How are you today? howis is"

# Obtener los tokens y sus IDs
tokens, token_ids = tokenize_sentence(input_sentence)

# Mostrar resultados
print("Frase de entrada:", input_sentence)
print("Tokens generados:", tokens)
print("IDs de tokens:", token_ids)

