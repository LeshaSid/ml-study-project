import streamlit as st 

st.set_page_config(
    page_title="ML Study Project",
    page_icon="🤖"
)

pages = [
    st.Page("pages/iris.py", title="Классификатор ирисов"),
    st.Page("pages/house_price_predict.py", title="Предсказатель стоимости жилья"),
    st.Page("pages/clients_clusterization.py", title="Кластеризация клиентов"),
    st.Page("pages/stock_price_forecast.py", title="Прогнозирование цены акций Spotify")
]

nav = st.navigation(pages, position="top")
nav.run()

