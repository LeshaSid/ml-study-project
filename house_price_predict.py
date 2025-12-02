import streamlit as st 
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def house_price_predict():
    house_price = fetch_california_housing()
    X = house_price.data
    y = house_price.target * 100000

    X_selected = X[:, :5]

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train, y_train)
    y_pred_linear = linear_regression_model.predict(X_test)
    r2_linear = r2_score(y_test, y_pred_linear)

    tree_model = DecisionTreeRegressor(random_state=42, max_depth=5)
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)
    r2_tree = r2_score(y_test, y_pred_tree)

    forest_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5) 
    forest_model.fit(X_train, y_train)
    y_pred_forest = forest_model.predict(X_test)
    r2_forest = r2_score(y_test, y_pred_forest)
    

    st.title("Предсказатель Стоимости Жилья 🏠")
    
    median_income = st.number_input("Средний доход в районе(млн $)", house_price.data[:, 0].min(), house_price.data[:, 0].max(), house_price.data[:, 0].mean())
    house_age = st.number_input("Возраст дома(лет)", house_price.data[:, 1].min(), house_price.data[:, 1].max(), house_price.data[:, 1].mean())
    ave_rooms = st.number_input("Среднее колличество комнат в доме(ед.)", house_price.data[:, 2].min(), house_price.data[:, 2].max(), house_price.data[:, 2].mean())
    ave_bedrooms = st.number_input("Среднее колличество спален в доме(ед.)", house_price.data[:, 3].min(), house_price.data[:, 3].max(), house_price.data[:, 3].mean())
    population = st.number_input("Население в одном блоке домов", house_price.data[:, 4].min(), house_price.data[:, 4].max(), house_price.data[:, 4].mean())

    features_df = np.array([[median_income, house_age, ave_rooms, ave_bedrooms, population]])
    prediction_linear = linear_regression_model.predict(features_df)
    prediction_tree = tree_model.predict(features_df)
    prediction_forest = forest_model.predict(features_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Стоимость жилья\n(линейная регрессия)", f"${prediction_linear[0]:,.0f}")
    with col2:
        st.metric("Стоимость жилья\n(решающее дерево)", f"${prediction_tree[0]:,.0f}")
    with col3:
        st.metric(
            "Стоимость жилья\n(случайный лес)", 
            f"${prediction_forest[0]:,.0f}", 
            delta=f"{(prediction_forest[0] - prediction_linear[0]) / 1000:,.1f}k",
            help="Ансамблевый метод, который снижает переобучение и шум."
        )

    st.divider()
    st.info(f"Линейная Регрессия (Baseline): **`{r2_linear:.4f}`**")
    st.info(f"Решающее Дерево (Max Depth 5): **`{r2_tree:.4f}`**")
    
    st.success(f"Случайный Лес (Random Forest): **`{r2_forest:.4f}`**")
    