from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

def chat_with_model(file_name):
    while True:
        # Read the entire content of the file
        with open(file_name, 'r') as file:
            previous_conversation = file.read()

        # Prepare the context string
        context = previous_conversation + "\n"

        # Get user input
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Prepare the full input for the model
        #full_input = context + f"Human: {user_input}\n"

        # Write the user's input to the file
        with open(file_name, 'a') as file:
            file.write(f"Human: {user_input}\n")

        # Tokenize the input text
        input_ids = tokenizer(user_input, return_tensors="pt").input_ids

        # Generate text using the model
        outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)

        # Decode the generated output
        assistant_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Print and write the model's response to the file
        print(f"Assistant: {assistant_response}")
        with open(file_name, 'a') as file:
            file.write(f"Assistant: {assistant_response}\n\n")

# Specify the filename where the conversation will be saved
file_name = 'text_file.txt'

# Start chatting
chat_with_model(file_name)
