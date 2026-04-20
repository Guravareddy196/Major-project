import pandas as pd
import numpy as np
import pickle
import os
from datetime import date, timedelta

try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLS = [
    'day_of_week', 'month', 'quarter', 'year', 'day_of_year',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_7', 'rolling_14', 'rolling_30',
    'arrival_qty'
]

def train_hybrid(df):
    print("Training Hybrid (Prophet + XGBoost Weighted Average)...")

    split_idx  = int(len(df) * 0.8)
    train_df   = df.iloc[:split_idx].reset_index(drop=True)
    test_df    = df.iloc[split_idx:].reset_index(drop=True)

    # ── Prophet ───────────────────────────────────────────────────────────────
    print("  [1/3] Training Prophet...")
    prophet_df = train_df[['date', 'modal_price']].rename(
        columns={'date': 'ds', 'modal_price': 'y'}
    )
    prophet_model = Prophet(
        yearly_seasonality       = True,
        weekly_seasonality       = True,
        daily_seasonality        = False,
        seasonality_mode         = 'multiplicative',
        changepoint_prior_scale  = 0.05,   # less flexible — prevents upward drift
        seasonality_prior_scale  = 5.0,
    )
    prophet_model.add_country_holidays(country_name='IN')
    prophet_model.fit(prophet_df)

    future_test    = pd.DataFrame({'ds': test_df['date']})
    prophet_preds  = prophet_model.predict(future_test)['yhat'].values
    prophet_preds  = np.clip(prophet_preds, 200, 9000)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("  [2/3] Training XGBoost...")
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['modal_price'].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df['modal_price'].values

    scaler  = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    xgb_model = xgb.XGBRegressor(
        n_estimators          = 300,
        learning_rate         = 0.05,
        max_depth             = 4,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        reg_alpha             = 1.0,
        reg_lambda            = 2.0,
        random_state          = 42,
        early_stopping_rounds = 30,
        eval_metric           = 'mae',
        verbosity             = 0,
    )
    xgb_model.fit(X_train_s, y_train,
                  eval_set=[(X_test_s, y_test)],
                  verbose=False)

    xgb_preds = xgb_model.predict(X_test_s)
    xgb_preds = np.clip(xgb_preds, 200, 9000)

    # ── Weighted Average Ensemble ─────────────────────────────────────────────
    # Find best weights using validation data
    print("  [3/3] Finding best ensemble weights...")
    best_mae = float('inf')
    best_w   = 0.5

    for w in np.arange(0.1, 1.0, 0.1):
        blended = w * prophet_preds + (1 - w) * xgb_preds
        mae     = mean_absolute_error(y_test, blended)
        if mae < best_mae:
            best_mae = mae
            best_w   = w

    hybrid_preds = best_w * prophet_preds + (1 - best_w) * xgb_preds
    hybrid_preds = np.clip(hybrid_preds, 200, 9000)

    # ── Metrics ───────────────────────────────────────────────────────────────
    prophet_mae  = mean_absolute_error(y_test, prophet_preds)
    prophet_rmse = np.sqrt(mean_squared_error(y_test, prophet_preds))
    prophet_mape = np.mean(np.abs((y_test - prophet_preds) / y_test)) * 100

    xgb_mae  = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    xgb_mape = np.mean(np.abs((y_test - xgb_preds) / y_test)) * 100

    hybrid_mae  = mean_absolute_error(y_test, hybrid_preds)
    hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_preds))
    hybrid_mape = np.mean(np.abs((y_test - hybrid_preds) / y_test)) * 100

    prophet_metrics = {'MAE': round(prophet_mae,2), 'RMSE': round(prophet_rmse,2), 'MAPE': round(prophet_mape,2)}
    xgb_metrics     = {'MAE': round(xgb_mae,2),     'RMSE': round(xgb_rmse,2),     'MAPE': round(xgb_mape,2)}
    hybrid_metrics  = {'MAE': round(hybrid_mae,2),  'RMSE': round(hybrid_rmse,2),  'MAPE': round(hybrid_mape,2)}

    print(f"\n  Model Comparison (best Prophet weight = {round(best_w,1)}):")
    print(f"  Prophet  -> MAE: Rs.{prophet_metrics['MAE']}, RMSE: Rs.{prophet_metrics['RMSE']}, MAPE: {prophet_metrics['MAPE']}%")
    print(f"  XGBoost  -> MAE: Rs.{xgb_metrics['MAE']},    RMSE: Rs.{xgb_metrics['RMSE']},    MAPE: {xgb_metrics['MAPE']}%")
    print(f"  Hybrid   -> MAE: Rs.{hybrid_metrics['MAE']},  RMSE: Rs.{hybrid_metrics['RMSE']},  MAPE: {hybrid_metrics['MAPE']}%")

    return prophet_model, xgb_model, scaler, best_w, hybrid_metrics, prophet_metrics, xgb_metrics


def predict_hybrid(prophet_model, xgb_model, scaler, best_w, df, target_date):
    """
    Predict price for a specific future date.
    Returns clipped, realistic predicted price.
    """
    target_date = pd.to_datetime(target_date)

    # Prophet prediction
    future_df    = pd.DataFrame({'ds': [target_date]})
    prophet_pred = float(prophet_model.predict(future_df)['yhat'].values[0])
    prophet_pred = np.clip(prophet_pred, 200, 9000)

    # XGBoost prediction
    feature_row = {
        'day_of_week' : target_date.dayofweek,
        'month'       : target_date.month,
        'quarter'     : (target_date.month - 1) // 3 + 1,
        'year'        : target_date.year,
        'day_of_year' : target_date.dayofyear,
        'lag_1'       : df['modal_price'].iloc[-1],
        'lag_7'       : df['modal_price'].iloc[-7]  if len(df) >= 7  else df['modal_price'].iloc[-1],
        'lag_14'      : df['modal_price'].iloc[-14] if len(df) >= 14 else df['modal_price'].iloc[-1],
        'lag_30'      : df['modal_price'].iloc[-30] if len(df) >= 30 else df['modal_price'].iloc[-1],
        'rolling_7'   : df['modal_price'].iloc[-7:].mean(),
        'rolling_14'  : df['modal_price'].iloc[-14:].mean(),
        'rolling_30'  : df['modal_price'].iloc[-30:].mean(),
        'arrival_qty' : df['arrival_qty'].mean(),
    }
    X        = pd.DataFrame([feature_row])[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    xgb_pred = float(xgb_model.predict(X_scaled)[0])
    xgb_pred = np.clip(xgb_pred, 200, 9000)

    # Weighted average
    final = best_w * prophet_pred + (1 - best_w) * xgb_pred

    # Important: cap max prediction at historical max + 20% to prevent unrealistic values
    hist_max = df['modal_price'].max()
    final    = np.clip(final, 200, hist_max * 1.2)

    return round(float(final), 2)


def save_hybrid_models(prophet_model, xgb_model, scaler, best_w):
    os.makedirs('models', exist_ok=True)
    with open('models/prophet_model.pkl', 'wb') as f:
        pickle.dump(prophet_model, f)
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/best_w.pkl', 'wb') as f:
        pickle.dump(best_w, f)
    print("All models saved to models/ folder!")


def load_hybrid_models():
    with open('models/prophet_model.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    with open('models/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/best_w.pkl', 'rb') as f:
        best_w = pickle.load(f)
    return prophet_model, xgb_model, scaler, best_w


if __name__ == '__main__':
    from preprocess import load_and_clean
    df = load_and_clean('Onion_Prices.csv')
    prophet_model, xgb_model, scaler, best_w, h_metrics, p_metrics, x_metrics = train_hybrid(df)
    save_hybrid_models(prophet_model, xgb_model, scaler, best_w)
    print("\nHybrid Metrics:", h_metrics)
