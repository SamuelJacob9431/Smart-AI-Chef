from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import json

# Load your dataset from the JSON file
with open("recipes.json", "r", encoding="utf-8") as f:
    recipes_data = json.load(f)

# Convert the list of recipes into a dataset with a "text" field
# Here, we combine the recipe fields into a single string for training.
def format_recipe(recipe):
    return (
        f"Title: {recipe['title']}\n"
        f"Ingredients: {', '.join(recipe['ingredients'])}\n"
        f"Prep Time: {recipe['prep_time']} minutes, Cook Time: {recipe['cook_time']} minutes\n"
        f"Servings: {recipe['servings']}\n"
        f"Instructions:\n{recipe['instructions']}\n"
    )

texts = [format_recipe(recipe) for recipe in recipes_data]

# Create a dataset in a simple format
dataset = {"text": texts}

# Save dataset temporarily to use the datasets library
import pandas as pd
df = pd.DataFrame(dataset)
df.to_csv("recipes_dataset.csv", index=False)

# Load the dataset using the datasets library
dataset = load_dataset("csv", data_files={"train": "recipes_dataset.csv"})["train"]

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-recipes-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
trainer.save_model("./gpt2-recipes-finetuned")
