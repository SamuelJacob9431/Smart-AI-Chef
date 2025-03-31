import pandas as pd

# Load ingredient stock and recipe database
stock_df = pd.read_csv('ingredient_stock.csv')
recipe_df = pd.read_csv('recipe_database.csv')
preference_df = pd.read_csv('customer_preference_scores.csv')   # <-- Load preferences here

# Make stock lookup
stock_dict = stock_df.set_index('ingredient').to_dict('index')

PROFIT_THRESHOLD = 0.2

data = []

for _, row in recipe_df.iterrows():
    ingredients = [i.strip() for i in row['ingredients'].split(",")]
    quantities = [float(q) for q in row['ingredient_quantity'].split(",")]
    dish_cost = 0
    expiry_days = []

    for ing, qty in zip(ingredients, quantities):
        if ing in stock_dict:
            ing_price = stock_dict[ing]['price_per_unit']
            dish_cost += qty * ing_price
            expiry_days.append(stock_dict[ing]['expiry_in_days'])
        else:
            dish_cost = None
            break

    if dish_cost is None:
        continue

    profit_margin = (row['selling_price'] - dish_cost) / row['selling_price']
    waste_score = 1 / (sum(expiry_days) / len(expiry_days) + 1e-5)

    # Get preference score
    pref_row = preference_df[preference_df['recipe_name'] == row['recipe_name']]
    preference_score = float(pref_row['preference_score']) if not pref_row.empty else 0

    label = 1 if profit_margin >= PROFIT_THRESHOLD else 0

    data.append({
        'recipe': row['recipe_name'],
        'dish_cost': round(dish_cost, 2),
        'selling_price': row['selling_price'],
        'profit_margin': round(profit_margin, 2),
        'waste_score': round(waste_score, 2),
        'preference_score': round(preference_score, 2),
        'label': label
    })

ai_dataset = pd.DataFrame(data)
ai_dataset.to_csv('ai_menu_training_data.csv', index=False)
print(ai_dataset)
