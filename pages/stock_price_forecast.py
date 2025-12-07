import streamlit as st 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Загрузка и обработка данных
try:
    df = pd.read_csv("World-Stock-Prices-Dataset.csv")
    df = df[df['Brand_Name'] == 'spotify']
    df = df[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
except FileNotFoundError:
    st.error("Файл не найден")
    st.stop()

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.sort_index()
df['Close'] = df['Close'].fillna(method='ffill')

# Создание признаков
df['Target_Price'] = df['Close'].shift(-1)
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['MA_7'] = df['Close'].rolling(window=7).mean().shift(1)
df = df.dropna()

# Подготовка данных
X = df[['Lag_1', 'Lag_2', 'MA_7', 'Volume']]
y = df[['Target_Price']]

test_size = int(0.3 * len(df))
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# Обучение моделей
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)

forest_model = RandomForestRegressor(max_depth=10, random_state=42)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
r2_forest = r2_score(y_test, y_pred_forest)

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Прогноз на 07.07.2025
real_price = 736.29
last_data = df.iloc[-1]

future_data = pd.DataFrame({
    'Lag_1': [last_data['Close']],
    'Lag_2': [last_data['Lag_1']],
    'MA_7': [last_data['MA_7']],
    'Volume': [last_data['Volume']]
})

future_price_linear = float(linear_model.predict(future_data)[0][0])
future_price_forest = float(forest_model.predict(future_data)[0])
future_price_xgb = float(xgb_model.predict(future_data)[0])

# Отображение результатов
st.title("Прогноз цены акций Spotify на 07.07.2025")

st.subheader("Прогнозы моделей на 07.07.2025")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Линейная регрессия", f"{future_price_linear:.2f}")

with col2:
    st.metric("Случайный лес", f"{future_price_forest:.2f}")

with col3:
    st.metric("XGBoost", f"{future_price_xgb:.2f}")

with col4:
    st.metric("Реальная цена", f"{real_price:.2f}")

st.divider()
st.subheader("Качество моделей на тестовых данных (R²)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Линейная регрессия", f"{r2_linear:.4f}")

with col2:
    st.metric("Случайный лес", f"{r2_forest:.4f}")

with col3:
    st.metric("XGBoost", f"{r2_xgb:.4f}")
