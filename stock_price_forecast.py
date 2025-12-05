import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def stock_price_forecast():
    try:
        df = pd.read_csv("World-Stock-Prices-Dataset.csv")
    except FileNotFoundError:
        st.error("Файл не найден")
        st.stop()

    df = df[df['Brand_Name'] == 'spotify']
    df = df[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    df['Close'] = df['Close'].fillna(method='ffill')

    df['Target_Difference'] = df['Close'].shift(-1) - df['Close']
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_Vol_1'] = df['Volume'].shift(1)
    df['Lag_High_1'] = df['High'].shift(1)
    df['Lag_Open_1'] = df['Open'].shift(1)
    df['Lag_Low_1'] = df['Low'].shift(1)
    df['Day_Of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Time_Step'] = np.arange(len(df))
    df = df.dropna()
    st.dataframe(df)
    
    X = df[['Lag_1', 'Lag_2', 'Lag_Vol_1', 'Lag_High_1', 'Lag_Open_1', 'Lag_Low_1', 'Day_Of_Week', 'Month', 'Time_Step']]
    y = df[['Target_Difference']]

    test_size = int(0.3 * len(df))
    X_train, X_test = X.iloc[test_size:], X.iloc[:test_size] 
    y_train, y_test = y.iloc[test_size:], y.iloc[:test_size]

    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train, y_train)
    y_pred_linear = linear_regression_model.predict(X_test)
    r2_linear = r2_score(y_test, y_pred_linear)

    random_forest_model = RandomForestRegressor(random_state=42, max_depth=10)
    random_forest_model.fit(X_train, y_train)
    y_pred_forest = random_forest_model.predict(X_test)
    r2_forest = r2_score(y_test, y_pred_forest)

    xgb_model = XGBRegressor()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    
    last_data_point = df.iloc[0]
    future_time_step = last_data_point['Time_Step'] + 1
    future_day_of_week = (last_data_point['Day_Of_Week'] + 1) % 5

    X_future_data = {
        'Lag_1': [last_data_point['Close']],
        'Lag_2': [last_data_point['Lag_1']],
        'Lag_Vol_1': [last_data_point['Volume']],
        'Lag_High_1': [last_data_point['High']],
        'Lag_Open_1': [last_data_point['Open']],
        'Lag_Low_1': [last_data_point['Low']],
        'Day_Of_Week': [future_day_of_week],
        'Month': [last_data_point['Month']],
        'Time_Step': [future_time_step]
    }
    
    X_future = pd.DataFrame(X_future_data, columns=X.columns)

    pred_diff_linear = linear_regression_model.predict(X_future)[0][0]
    pred_diff_forest = random_forest_model.predict(X_future)[0]
    pred_diff_xgb = xgb_model.predict(X_future)[0]
    
    last_known_close = last_data_point['Close']
    
    future_pred_linear = last_known_close + pred_diff_linear
    future_pred_forest = last_known_close + pred_diff_forest
    future_pred_xgb = last_known_close + pred_diff_xgb
    
    last_known_close = last_data_point['Close']
    st.title("Прогноз цены акций Spotify на 03.07.2025")
    st.metric("Цена закрытия 02.07.2025", last_known_close)

    st.subheader("Прогноз цены на 02.07.2025")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Линейная регрессия", future_pred_linear)
    with col2:
        st.metric("Случайный лес", future_pred_forest)
    with col3:
        st.metric("XGBoost регрессия", future_pred_xgb)

    st.metric("Реальная цена на 03.07.2025", 725.05)

    st.divider()

    st.info("Метрики $R^2$ на тестовой выборке") 
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Линейная регрессия", f"{r2_linear:.4f}")
    with col2:
        st.metric("Случайный лес", f"{r2_forest:.4f}")
    with col3:
        st.metric("XGBoost регрессия", f"{r2_xgb:.4f}")