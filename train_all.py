"""
train_all.py - Run this ONCE to train all models.
After this, run app.py to start the dashboard.
"""
from preprocess import load_and_clean
from hybrid_model import train_hybrid, save_hybrid_models, predict_hybrid
from datetime import date, timedelta

print("=" * 55)
print("  ONION PRICE PREDICTION - MODEL TRAINING")
print("=" * 55)

print("\n[Step 1/2] Preprocessing data...")
df = load_and_clean('Onion_Prices.csv')
df.to_csv('cleaned_onion_data.csv', index=False)
print("  Saved: cleaned_onion_data.csv")

print("\n[Step 2/2] Training Hybrid Model (Prophet + XGBoost)...")
prophet_model, xgb_model, scaler, best_w, h_metrics, p_metrics, x_metrics = train_hybrid(df)
save_hybrid_models(prophet_model, xgb_model, scaler, best_w)

# Quick test predictions
print("\nTest predictions:")
for days in [7, 30, 90]:
    test_date  = date.today() + timedelta(days=days)
    test_price = predict_hybrid(prophet_model, xgb_model, scaler, best_w, df, str(test_date))
    print(f"  {days:3d} days from today ({test_date}): Rs.{test_price}/quintal  (Rs.{round(test_price/100,2)}/kg)")

print("\n" + "=" * 55)
print("  TRAINING COMPLETE!")
print(f"  Prophet MAE  : Rs.{p_metrics['MAE']}  |  MAPE: {p_metrics['MAPE']}%")
print(f"  XGBoost MAE  : Rs.{x_metrics['MAE']}  |  MAPE: {x_metrics['MAPE']}%")
print(f"  Hybrid  MAE  : Rs.{h_metrics['MAE']}  |  MAPE: {h_metrics['MAPE']}%")
print("=" * 55)
print("\n  Now run:  python app.py")
print("  Then open: http://localhost:5000")
print("=" * 55)
