# 🧅 Onion Price Prediction Dashboard
### B.Tech Major Project — 4-2 Semester
**Market:** Kurnool APMC, Andhra Pradesh  
**Models:** Prophet + XGBoost Hybrid

---

## 📁 Project Structure

```
onion_project/
│
├── Onion_Prices.csv          ← Your dataset (place it here)
├── preprocess.py             ← Data cleaning & feature engineering
├── prophet_model.py          ← Facebook Prophet model
├── xgboost_model.py          ← XGBoost model
├── hybrid_model.py           ← Hybrid (Prophet + XGBoost residual correction)
├── train_all.py              ← Run this ONCE to train all models
├── app.py                    ← Flask web dashboard
├── requirements.txt          ← Python dependencies
│
├── templates/
│   └── index.html            ← Dashboard UI
│
└── models/                   ← Saved models (auto-created after training)
    ├── prophet_model.pkl
    ├── residual_xgb.pkl
    └── hybrid_scaler.pkl
```

---

## ⚙️ Setup Instructions

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Place your dataset
Copy `Onion_Prices.csv` into the project folder.

### Step 3 — Train the models (run ONCE)
```bash
python train_all.py
```

### Step 4 — Start the dashboard
```bash
python app.py
```

### Step 5 — Open in browser
```
http://localhost:5000
```

---

## 🧠 How the Hybrid Model Works

```
Raw Data
   ↓
Preprocessing (clean, fill missing dates, feature engineering)
   ↓
Prophet Model → captures trend + seasonality → base prediction
   ↓
XGBoost Model → trained on Prophet's residuals → correction
   ↓
Final Price = Prophet Prediction + XGBoost Correction
```

---

## 📊 Dashboard Features

- **SELL button** → Shows current market price and today's earnings
- **KEEP button** → Shows predicted price on target date, profit or loss
- **Price trend chart** → Historical 30 days + 30-day forecast
- **Stats panel** → Current price, predicted price, earnings comparison

---

## 📈 Metrics

| Metric | Description |
|--------|-------------|
| MAE    | Mean Absolute Error (average Rs. error per prediction) |
| RMSE   | Root Mean Squared Error |
| MAPE   | Mean Absolute Percentage Error (% accuracy) |
