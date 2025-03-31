from transformers import pipeline

def generate_daily_special(ingredients):
    # Create a comma-separated string of ingredients.
    ingredients_str = ", ".join(ingredients)
    
    # Build a shorter, more direct prompt.
    prompt = (
        f"Create a detailed recipe using these ingredients: {ingredients_str}. "
        "Output only a numbered list containing: \n"
        "1. Ingredients with exact measurements, \n"
        "2. Step-by-step cooking instructions, \n"
        "3. Preparation and cooking times (in minutes)."
    )
    
    # Initialize the text-generation pipeline with sampling enabled.
    generator = pipeline(
        "text-generation",
        model="gpt2",
        do_sample=True,       # Enable sampling for more creative outputs.
        max_length=400,       # Increase maximum output length.
        temperature=0.9,      # Adjust randomness.
        top_k=50,             # Top-k sampling.
        top_p=0.95            # Nucleus sampling.
    )
    
    # Generate the recipe suggestion.
    result = generator(prompt, max_length=400, truncation=True, num_return_sequences=1)
    return result[0]["generated_text"]

if __name__ == "__main__":
    # Ask the user for ingredients.
    user_input = input("Enter surplus or soon-to-expire items separated by commas: ")
    ingredients_list = [item.strip() for item in user_input.split(",")]
    
    # Generate and print the daily special recipe.
    recipe = generate_daily_special(ingredients_list)
    print("\nDaily Special Recipe:")
    print(recipe)
