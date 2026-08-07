from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris
iris=load_iris()

x= iris.data
y= iris.target
model=RandomForestClassifier()
model.fit(x,y)

st.title("simple iris classifier")

sepal_length=st.number_input("sepal length",min_value=0.0,max_value=10.0,value=5.0)
sepal_width=st.number_input("sepal width",min_value=0.0,max_value=10.0,value=3.0)
petal_length=st.number_input("petal length",min_value=0.0,max_value=10.0,value=1.0)
petal_width=st.number_input("petal width",min_value=0.0,max_value=10.0,value=0.2)
predict=st.button("predict")

if predict:
    input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction=model.predict(input_data)
    species=iris.target_names[prediction][0]
    st.success(f"The predicted species is: {species}")
