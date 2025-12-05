import streamlit as st 
from iris import iris
from house_price_predict import house_price_predict
from clients_clusterization import client_clusterization
from stock_price_forecast import stock_price_forecast

st.set_page_config(
    page_title="ML Study Project",
    page_icon="🤖"
)

tab1, tab2, tab3, tab4 = st.tabs(["Классификатор Ирисов 🌸", "Предсказатель Стоимости Жилья 🏠", "Кластеризация клиентов", "Прогнозирование цены акций Spotify"])

with tab1:
    iris()
with tab2:
    house_price_predict()
with tab3:
    client_clusterization()
with tab4:
    stock_price_forecast()