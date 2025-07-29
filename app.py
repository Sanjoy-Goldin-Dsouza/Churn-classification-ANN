import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import tensorflow as tf

st.title('Customer Churn Prediction')

# load Encoder
label_encoder_gender=pickle.load(open('label_encoder_gender.pkl','rb'))
one_hot_encoder_geo=pickle.load(open('one_hot_encoder_geo.pkl','rb'))
standard=pickle.load(open('standard.pkl','rb'))

# User input
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}, index=[0])

geo_encoded = one_hot_encoder_geo.transform([input_data['Geography']]).toarray()
one_hot_geo_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
test_df=pd.concat([input_data.drop(['Geography'],axis=1),one_hot_geo_df],axis=1)
test_df=standard.transform(test_df)

#load model
model=tf.keras.models.load_model('model.keras')

predict_value=model.predict(test_df)

if st.button('Submit'):
    if predict_value[0][0]>0.5:
        st.write('Customer will join')
    else:
        st.write('Customer will not join')




