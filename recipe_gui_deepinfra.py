import tkinter as tk
from tkinter import messagebox
import requests

# ✅ DeepInfra API Token
API_TOKEN = "zKS4a79d7Ei2fVzeEV5Ytr346Yqe5QQG"  # <-- put your token here

def generate_recipe():
    ingredients = entry.get()
    if not ingredients.strip():
        messagebox.showerror("Error", "Please enter at least one ingredient.")
        return

    # Prompt
    prompt = f"""
You are a professional recipe generator.

TASK: Create a detailed step-by-step cooking recipe using the following ingredients: {ingredients}

STRICT FORMAT:
1. Ingredients section (with exact quantities)
2. Step-by-step Instructions (numbered and very detailed)
3. Preparation time (in minutes)
4. Cooking time (in minutes)

DO NOT include tips, nutritional info, or comments. Only the recipe.
"""


    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }

    payload = {
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "messages": [{"role": "user", "content": prompt}]
}


    response = requests.post("https://api.deepinfra.com/v1/openai/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        recipe_output.delete("1.0", tk.END)
        recipe_output.insert(tk.END, result['choices'][0]['message']['content'])
    else:
        messagebox.showerror("Error", f"API Error: {response.text}")

# -------- GUI ---------
root = tk.Tk()
root.title("DeepInfra AI Recipe Generator")

tk.Label(root, text="Enter random ingredients (comma-separated):").pack(pady=5)
entry = tk.Entry(root, width=60)
entry.pack(pady=5)

tk.Button(root, text="Generate Recipe", command=generate_recipe).pack(pady=10)

recipe_output = tk.Text(root, wrap=tk.WORD, width=70, height=20)
recipe_output.pack(padx=10, pady=10)

root.mainloop()
