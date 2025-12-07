import streamlit as st 
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

target_names = iris.target_names

st.title("Классификатор Ирисов 🌸")
sepal_length = st.slider("Длина чашелистика", iris.data[:, 0].min(), iris.data[:, 0].max(), iris.data[:, 0].mean())
sepal_width = st.slider("Ширина чашелистика", iris.data[:, 1].min(), iris.data[:, 1].max(), iris.data[:, 1].mean())
petal_length = st.slider("Длинна лепестка", iris.data[:, 2].min(), iris.data[:, 2].max(), iris.data[:, 2].mean())
petal_width = st.slider("Ширина лепестка", iris.data[:, 3].min(), iris.data[:, 3].max(), iris.data[:, 3].mean())

data = {
    "Длина чашелистика(см)" : sepal_length,
    "Ширина чашелистика(см)" : sepal_width,
    "Длинна лепестка(см)" : petal_length,
    "Ширина лепестка(см)" : petal_width
}

features_df = pd.DataFrame(data, index=[0])

prediction = model.predict(features_df)
prediction_proba = model.predict_proba(features_df)

st.subheader("Предсказание вида ириса: ")
st.success(f"Вид **{target_names[prediction][0]}**")
if target_names[prediction][0] == "setosa":
    st.image("images/setosa.jpg")
if target_names[prediction][0] == "versicolor":
    st.image("images/versicolor.jpg")
if target_names[prediction][0] == "virginica":
    st.image("images/virginica.jpg")

st.subheader("Вероятности классов: ")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.dataframe(proba_df, hide_index=True)

st.divider()
st.write(f"Точность модели на тесте: **{accuracy:.2f}**")






