import streamlit as st 
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

def client_clusterization():
    try:
        df = pd.read_csv("Mall_Customers.csv")
    except FileNotFoundError:
        st.error("Файл не найден!")
        st.stop()

    X = df.iloc[:, [3, 4]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.title("Кластеризация покупателей")
    K_slider = st.slider("Выберите количество кластеров(К)", 2, 10, 5)

    kmeans = KMeans(n_clusters=K_slider, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    original_centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    st.subheader("Результаты кластеризации")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, alpha=0.8)
    ax.scatter(original_centroids[:, 0], original_centroids[:, 1], marker='X', s=300, c="red", label="Центроиды")
    ax.set_title("Кластеризация клиентов")
    ax.set_xlabel("Годовой доход(тыс. $)")
    ax.set_ylabel("Оценка трат(1-100)")
    ax.legend()
    plt.grid(True)

    st.pyplot(fig)

    st.divider()
    st.subheader("Интерпретация результатов")
    st.html(
        """
        <h3><span style="color:yellow">1. Низкий доход / низкое потребление</span></h3>
        Бедные клиенты, потребляющие мало
        <h3><span style="color:MediumTurquoise">2. Низкий доход / высокое потребление</span></h3>
        Клиенты, живущие в кредит
        <h3><span style="color:violet">3. Средний доход / среднее потребление</span></h3>
        Средний класс, живущие по средствам
        <h3><span style="color:lightgreen">4. Высокий доход / низкое потребление</span></h3>
        Богатые бережливые клиенты
        <h3><span style="color:SteelBlue">5. Высокий доход / высокое потребление</span></h3>
        Богатые клиенты с большим потреблением
        """
    )


