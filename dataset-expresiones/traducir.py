from transformers import MarianMTModel, MarianTokenizer

# Cargar el modelo y el tokenizador de MarianMT para traducción de inglés a español
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_sentence(sentence):
    # Tokenizar la entrada
    tokenized_input = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # Realizar la traducción
    translated = model.generate(**tokenized_input)

    # Decodificar los tokens de salida
    translated_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_sentence

# Bucle interactivo para tomar entradas del usuario
print("Introduce la frase en inglés que deseas traducir (escribe 'exit' para salir):")

# Abrir el archivo en modo escritura
with open("traducciones.txt", "w", encoding="utf-8") as file:
    while True:
        input_sentence = input("Frase en inglés: ")
        if input_sentence.lower() == "exit":
            print("Saliendo...")
            break  # Salir del bucle si el usuario escribe "exit"

        # Obtener la traducción
        translated_sentence = translate_sentence(input_sentence)

        # Mostrar resultados
        print("Traducción al español:", translated_sentence)

        # Guardar la interacción en el archivo
        file.write(f"Inglés: {input_sentence}\n")
        file.write(f"Español: {translated_sentence}\n\n")
        
        # Forzar la escritura en el archivo
        file.flush()

print("Traducciones guardadas en traducciones.txt.")

