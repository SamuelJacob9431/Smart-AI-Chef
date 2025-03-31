from transformers import pipeline

# Initialize a text-generation pipeline using GPT-2
generator = pipeline("text-generation", model="datificate/gpt2-small-recipe-generator")


# Define a prompt listing available ingredients
prompt = "With the ingredients tomato, cheese, basil, and garlic, create a unique recipe for a delicious meal:"

# Generate a recipe suggestion
result = generator(prompt, max_length=100, num_return_sequences=1)
print("Recipe Suggestion:")
print(result[0]['generated_text'])
