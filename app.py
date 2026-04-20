from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import date, datetime
import os

from preprocess import load_and_clean
from hybrid_model import predict_hybrid, load_hybrid_models

app = Flask(__name__)

DF             = None
PROPHET_MODEL  = None
XGB_MODEL      = None
SCALER         = None
BEST_W         = None
MARKETS        = ['Kurnool APMC', 'Pattikonda APMC']


def initialize():
    global DF, PROPHET_MODEL, XGB_MODEL, SCALER, BEST_W
    print("Loading data and models...")
    DF = load_and_clean('Onion_Prices.csv')
    if os.path.exists('models/prophet_model.pkl'):
        PROPHET_MODEL, XGB_MODEL, SCALER, BEST_W = load_hybrid_models()
        print("Models loaded successfully!")
    else:
        print("Models not found. Run train_all.py first.")


@app.route('/')
def index():
    today = date.today().strftime('%Y-%m-%d')
    return render_template('index.html', markets=MARKETS, today=today)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data        = request.get_json()
        action      = data.get('action')
        target_date = data.get('target_date')
        quantity_kg = float(data.get('quantity_kg', 100))
        market      = data.get('market', 'Kurnool APMC')

        # Current price = latest price from dataset
        current_price_quintal = float(DF['modal_price'].iloc[-1])
        current_price_kg      = round(current_price_quintal / 100, 2)
        current_earnings      = round(current_price_kg * quantity_kg, 2)

        # Predicted price on target date
        predicted_price_quintal = predict_hybrid(
            PROPHET_MODEL, XGB_MODEL, SCALER, BEST_W, DF, target_date
        )
        predicted_price_kg  = round(predicted_price_quintal / 100, 2)
        predicted_earnings  = round(predicted_price_kg * quantity_kg, 2)
        difference          = round(predicted_earnings - current_earnings, 2)

        # Profit or Loss determination
        if difference > 0:
            recommendation = 'profit'
            diff_text      = f"a PROFIT of Rs.{difference:.2f}"
        elif difference < 0:
            recommendation = 'loss'
            diff_text      = f"a LOSS of Rs.{abs(difference):.2f}"
        else:
            recommendation = 'neutral'
            diff_text      = "no change"

        if action == 'sell':
            recommendation = 'sell'
            message = (
                f"Current market price at {market} is Rs.{current_price_kg}/kg "
                f"(Rs.{current_price_quintal}/quintal). "
                f"For {quantity_kg} kg, you will earn Rs.{current_earnings:.2f} today."
            )
        else:
            message = (
                f"Predicted price on {target_date} at {market} is Rs.{predicted_price_kg}/kg "
                f"(Rs.{predicted_price_quintal}/quintal). "
                f"For {quantity_kg} kg, you will earn Rs.{predicted_earnings:.2f}. "
                f"That is {diff_text} compared to selling today."
            )

        # Price trend for chart (last 30 days + next 30 days forecast)
        last_30 = DF[['date', 'modal_price']].tail(30).copy()
        last_30['date'] = last_30['date'].dt.strftime('%Y-%m-%d')

        future_dates  = pd.date_range(start=date.today(), periods=30, freq='D')
        future_prices = []
        for fd in future_dates:
            p = predict_hybrid(PROPHET_MODEL, XGB_MODEL, SCALER, BEST_W, DF, fd)
            future_prices.append({'date': fd.strftime('%Y-%m-%d'), 'modal_price': p})

        chart_data = last_30.to_dict('records') + future_prices

        return jsonify({
            'success'            : True,
            'action'             : action,
            'message'            : message,
            'recommendation'     : recommendation,
            'current_price_kg'   : current_price_kg,
            'predicted_price_kg' : predicted_price_kg,
            'current_earnings'   : current_earnings,
            'predicted_earnings' : predicted_earnings,
            'difference'         : difference,
            'chart_data'         : chart_data,
            'target_date'        : target_date,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    initialize()
    app.run(debug=True, port=5000)
