import pandas as pd
import joblib

# Load AI Dataset (already includes preference_score)
data = pd.read_csv('ai_menu_training_data.csv')

# Load AI Model
model = joblib.load('menu_ai_model.pkl')

# Prepare features
X = data[['dish_cost', 'profit_margin', 'waste_score', 'preference_score']]

# Predict suitability
data['predicted'] = model.predict(X)

# Display optimized menu
print("\n🔵 AI Optimized Menu (Considering Profit + Waste + Customer Preference)\n")

recommended = data[data['predicted'] == 1]

if recommended.empty:
    print("⚠ No dishes meet the conditions today.")
else:
    print(f"\n{'Dish':30} | {'Dish Cost':>10} | {'Selling Price':>14} | {'Profit ₹':>9} | {'Profit %':>9} | {'Waste Score':>12} | {'Preference':>11}")
    print("-" * 110)
    for _, row in recommended.iterrows():
        profit_amount = row['selling_price'] - row['dish_cost']
        print(f"{row['recipe'][:30]:30} | ₹{row['dish_cost']:8.2f} | ₹{row['selling_price']:12.2f} | ₹{profit_amount:7.2f} | {row['profit_margin']*100:7.2f}% | {round(row['waste_score'],2):11} | {round(row['preference_score'],2):10}")
