import pandas as pd
import numpy as np
import os

def load_and_clean(filepath='Onion_Prices.csv'):
    df = pd.read_csv(filepath, header=1)
    df = df[df['State'] == 'Andhra Pradesh'].copy()
    df = df.rename(columns={
        'Date': 'date',
        'Market': 'market',
        'District': 'district',
        'Modal Price 23-01-2021 to 23-03-2026': 'modal_price',
        'Arrival Quantity 23-01-2021 to 23-03-2026': 'arrival_qty',
    })
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df = df[df['market'] == 'Kurnool APMC'][['date', 'modal_price', 'arrival_qty']].copy()
    df = df.sort_values('date').reset_index(drop=True)
    df = df.groupby('date').agg({'modal_price': 'mean', 'arrival_qty': 'sum'}).reset_index()

    full_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    df = df.set_index('date').reindex(full_range)
    df.index.name = 'date'
    df['modal_price'] = df['modal_price'].ffill()
    df['arrival_qty'] = df['arrival_qty'].fillna(0)
    df = df.reset_index()
    df = df[(df['modal_price'] >= 200) & (df['modal_price'] <= 9000)]

    df['day_of_week']  = df['date'].dt.dayofweek
    df['month']        = df['date'].dt.month
    df['quarter']      = df['date'].dt.quarter
    df['year']         = df['date'].dt.year
    df['day_of_year']  = df['date'].dt.dayofyear
    df['lag_1']        = df['modal_price'].shift(1)
    df['lag_7']        = df['modal_price'].shift(7)
    df['lag_14']       = df['modal_price'].shift(14)
    df['lag_30']       = df['modal_price'].shift(30)
    df['rolling_7']    = df['modal_price'].shift(1).rolling(7).mean()
    df['rolling_14']   = df['modal_price'].shift(1).rolling(14).mean()
    df['rolling_30']   = df['modal_price'].shift(1).rolling(30).mean()
    df = df.dropna().reset_index(drop=True)

    print(f"Preprocessing complete!")
    print(f"   Total rows    : {len(df)}")
    print(f"   Date range    : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Price range   : Rs.{df['modal_price'].min():.0f} - Rs.{df['modal_price'].max():.0f} per quintal")
    print(f"   Average price : Rs.{df['modal_price'].mean():.0f} per quintal")
    return df

if __name__ == '__main__':
    df = load_and_clean('Onion_Prices.csv')
    df.to_csv('cleaned_onion_data.csv', index=False)
    print("Saved as cleaned_onion_data.csv")
