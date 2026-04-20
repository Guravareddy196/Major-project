import pandas as pd
import numpy as np
import pickle
import os
from preprocess import load_and_clean

# Prophet import
try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet

from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_prophet(df):
    """
    Train a Prophet model on the cleaned onion price data.
    Returns: trained model, forecast dataframe, metrics dict
    """

    print("🔮 Training Prophet model...")

    # Prophet requires columns: ds (date) and y (target)
    prophet_df = df[['date', 'modal_price']].rename(
        columns={'date': 'ds', 'modal_price': 'y'}
    )

    # ── Train/Test Split (80/20) ──────────────────────────────────────────────
    split_idx   = int(len(prophet_df) * 0.8)
    train_df    = prophet_df.iloc[:split_idx]
    test_df     = prophet_df.iloc[split_idx:]

    # ── Train Prophet ─────────────────────────────────────────────────────────
    model = Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = True,
        daily_seasonality   = False,
        seasonality_mode    = 'multiplicative',   # better for price data
        changepoint_prior_scale = 0.1,            # flexibility of trend
    )

    # Add Indian festival effects as holidays
    indian_festivals = pd.DataFrame({
        'holiday'   : 'Indian_Festival',
        'ds'        : pd.to_datetime([
            '2021-11-04', '2022-10-24', '2023-11-13', '2024-11-01', '2025-10-20',  # Diwali approx
            '2021-10-15', '2022-10-05', '2023-10-24', '2024-10-12', '2025-10-02',  # Navratri approx
        ]),
        'lower_window': -2,
        'upper_window':  2,
    })
    model.add_country_holidays(country_name='IN')

    model.fit(train_df)

    # ── Predict on test period ────────────────────────────────────────────────
    future      = model.make_future_dataframe(periods=len(test_df), freq='D')
    forecast    = model.predict(future)

    # Align predictions with actual test values
    test_preds  = forecast.iloc[split_idx:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    test_actual = test_df.reset_index(drop=True)

    y_true = test_actual['y'].values
    y_pred = test_preds['yhat'].values[:len(y_true)]

    # Clip negative predictions
    y_pred = np.clip(y_pred, 0, None)

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}

    print(f"   ✅ Prophet Training complete!")
    print(f"   MAE  : ₹{metrics['MAE']}")
    print(f"   RMSE : ₹{metrics['RMSE']}")
    print(f"   MAPE : {metrics['MAPE']}%")

    # ── Save residuals for hybrid model ──────────────────────────────────────
    residuals = y_true - y_pred
    residual_df = test_actual.copy()
    residual_df['prophet_pred'] = y_pred
    residual_df['residual']     = residuals

    return model, forecast, metrics, residual_df


def predict_prophet(model, future_date):
    """
    Predict price for a specific future date using trained Prophet model.
    Returns predicted price (float).
    """
    future_df = pd.DataFrame({'ds': [pd.to_datetime(future_date)]})
    forecast  = model.predict(future_df)
    predicted = float(forecast['yhat'].values[0])
    predicted = max(0, predicted)  # no negative prices
    return round(predicted, 2)


def save_prophet_model(model, path='models/prophet_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"💾 Prophet model saved to {path}")


def load_prophet_model(path='models/prophet_model.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    df = load_and_clean('Onion_Prices.csv')
    model, forecast, metrics, residuals = train_prophet(df)
    save_prophet_model(model)
    residuals.to_csv('models/prophet_residuals.csv', index=False)
    print("\n📊 Prophet Metrics:", metrics)
