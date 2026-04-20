import pandas as pd
import numpy as np
import pickle
import os
from preprocess import load_and_clean

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Features used for XGBoost
FEATURE_COLS = [
    'day_of_week', 'month', 'quarter', 'year', 'day_of_year',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_7', 'rolling_14', 'rolling_30',
    'arrival_qty'
]
TARGET_COL = 'modal_price'


def train_xgboost(df):
    """
    Train an XGBoost model on the cleaned onion price data.
    Returns: trained model, scaler, metrics dict
    """

    print("🚀 Training XGBoost model...")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # ── Train/Test Split (80/20) ──────────────────────────────────────────────
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── Scale features ────────────────────────────────────────────────────────
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    model = xgb.XGBRegressor(
        n_estimators        = 500,
        learning_rate       = 0.05,
        max_depth           = 6,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        reg_alpha           = 0.1,
        reg_lambda          = 1.0,
        random_state        = 42,
        early_stopping_rounds = 50,
        eval_metric         = 'rmse',
        verbosity           = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set    = [(X_test, y_test)],
        verbose     = False
    )

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics = {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}

    print(f"   ✅ XGBoost Training complete!")
    print(f"   MAE  : ₹{metrics['MAE']}")
    print(f"   RMSE : ₹{metrics['RMSE']}")
    print(f"   MAPE : {metrics['MAPE']}%")

    return model, scaler, metrics


def predict_xgboost(model, scaler, df, target_date):
    """
    Predict price for a specific future date using XGBoost.
    Uses the last row of df to build lag/rolling features.
    Returns predicted price (float).
    """
    target_date = pd.to_datetime(target_date)
    last_row    = df.iloc[-1]

    # Build feature row for target date
    feature_row = {
        'day_of_week' : target_date.dayofweek,
        'month'       : target_date.month,
        'quarter'     : (target_date.month - 1) // 3 + 1,
        'year'        : target_date.year,
        'day_of_year' : target_date.dayofyear,
        'lag_1'       : last_row['modal_price'],
        'lag_7'       : df['modal_price'].iloc[-7]  if len(df) >= 7  else last_row['modal_price'],
        'lag_14'      : df['modal_price'].iloc[-14] if len(df) >= 14 else last_row['modal_price'],
        'lag_30'      : df['modal_price'].iloc[-30] if len(df) >= 30 else last_row['modal_price'],
        'rolling_7'   : df['modal_price'].iloc[-7:].mean(),
        'rolling_14'  : df['modal_price'].iloc[-14:].mean(),
        'rolling_30'  : df['modal_price'].iloc[-30:].mean(),
        'arrival_qty' : df['arrival_qty'].mean(),
    }

    X = pd.DataFrame([feature_row])[FEATURE_COLS].values
    X = scaler.transform(X)

    predicted = float(model.predict(X)[0])
    predicted = max(0, predicted)
    return round(predicted, 2)


def save_xgboost_model(model, scaler, model_path='models/xgboost_model.pkl', scaler_path='models/xgb_scaler.pkl'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 XGBoost model saved to {model_path}")


def load_xgboost_model(model_path='models/xgboost_model.pkl', scaler_path='models/xgb_scaler.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


if __name__ == '__main__':
    df = load_and_clean('Onion_Prices.csv')
    model, scaler, metrics = train_xgboost(df)
    save_xgboost_model(model, scaler)
    print("\n📊 XGBoost Metrics:", metrics)
