import pandas as pd

# Load ingredient stock
stock_df = pd.read_csv('ingredient_stock.csv')

# Load recipe database
recipe_df = pd.read_csv('recipe_database.csv')

# Convert stock to dictionary for easy lookup
stock_dict = stock_df.set_index('ingredient').to_dict('index')

# Helper function to check if a recipe can be made
def is_recipe_feasible(recipe_ingredients, recipe_quantities):
    recipe_ingredients = str(recipe_ingredients)
    recipe_quantities = str(recipe_quantities)
    
    ingredients = [i.strip() for i in recipe_ingredients.split(",")]
    quantities = [float(q) for q in recipe_quantities.split(",")]

    for ing, qty in zip(ingredients, quantities):
        if ing not in stock_dict:
            return False
        if stock_dict[ing]['available_quantity'] < qty:
            return False
        if stock_dict[ing]['expiry_in_days'] <= 0:
            return False
    return True


# Build optimized menu
optimized_menu = []

for index, row in recipe_df.iterrows():
    if is_recipe_feasible(row['ingredients'], row['ingredient_quantity']):
        optimized_menu.append(row['recipe_name'])

# Show results
print("🔵 Optimized Menu for Today 🔵")
if optimized_menu:
    for dish in optimized_menu:
        print(f"- {dish}")
else:
    print("No suitable dishes can be prepared with current stock.")
