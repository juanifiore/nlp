from transformers import MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-es'

tokenizer = MarianTokenizer.from_pretrained(model_name)

num_tokens = len(tokenizer)
print(num_tokens)
