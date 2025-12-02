import streamlit as st 
from iris import iris
from house_price_predict import house_price_predict

tab1, tab2 = st.tabs(["Классификатор Ирисов 🌸", "Предсказатель Стоимости Жилья 🏠"])

with tab1:
    iris()
with tab2:
    house_price_predict()