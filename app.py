import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import pickle

model=tf.keras.models.load_model('model.h5')

with open('geo_encoder.pkl','rb') as file:
    geo_encoder=pickle.load(file)

with open('gender_encoder.pkl','rb') as file:
    gender_encoder=pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
st.title("Bank Customer Churn Prediction")

geography=st.selectbox('Geography',geo_encoder.categories_[0])
gender=st.selectbox('Gender',gender_encoder.classes_)
age=st.slider("Age",18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider("Number of Products",1,4)
has_cr_cd=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is Active Member",[0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_cd],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded=geo_encoder.transform([input_data['Geography']]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=geo_encoder.get_feature_names_out(['Geography']))
input_df=pd.DataFrame(input_data)
input_df=pd.concat([input_df.reset_index(drop=True),geo_encoded_df],axis=1)
input_df=input_df.drop('Geography',axis=1)
input_df['Gender']=gender_encoder.transform(input_df['Gender'])
input_scaled=scaler.transform(input_df)

prediction=model.predict(input_scaled)
pred_prob=prediction[0][0]
st.write(f"The churn probability is: {pred_prob}")
if(pred_prob<0.5):
    st.write("The customer is not likely to churn.")
else:
    st.write("The customer is likely to churn.")