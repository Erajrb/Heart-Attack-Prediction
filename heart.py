import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import os
os.chdir("/Users/eraj/Desktop/Projects/HeartDisease")
print("Current working directory:", os.getcwd())

st.title("Heart Disease Predictor")
tab1,tab2 = st.tabs(['Predict','Model Information'])

# Create input 
with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex",["Male","Female"])
    chest_pain = st.selectbox("Chest Paint Type",["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic Angina"])
    resting_bp = st.number_input("Resting Blood Pressure",min_value=0,max_value=300)
    cholesterol = st.number_input("Serum Cholesterol",min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar",["<= 120 mg/dl",">120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results",["Normal","ST-T wave abnormality","Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Max Heart Rate Achieved",min_value=60,max_value=250)
    exercise_angina = st.selectbox("Exercise induced angina",["Yes","No"])
    oldpeak = st.number_input("Oldpeak(ST Depression)",min_value=0.0,max_value=10.0)
    st_slope = st.selectbox("Slope of peak exercise ST segment",["Upsloping","Flat","Downsloping"])

# Convert categorical values to numerical
sex = 0 if sex == "Male" else 1
chest_pain = ["Atypical Angina","Non-Anginal Pain","Asymptomatic Angina","Typical Angina"].index(chest_pain)
fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
resting_ecg = ["Normal","ST-T wave abnormality","Left Ventricular Hypertrophy"].index(resting_ecg)
exercise_angina = 1 if exercise_angina == "Yes"else 0
st_slope = ["Upsloping","Flat","Downsloping"].index(st_slope)

# Create a dataframe with user inputs
input_data = pd.DataFrame({
    'Age':[age],
    'Sex':[sex],
    'ChestPainType':[chest_pain],
    'RestingBP':[resting_bp],
    'Cholesterol':[cholesterol],
    'FastingBS':[fasting_bs],
    'RestingECG':[resting_ecg],
    'MaxHR':[max_hr],
    'ExerciseAngina':[exercise_angina],
    'Oldpeak':[oldpeak],
    'ST_Slope':[st_slope]
})
 
# Load the models
algonames =['Decision Tree','Logistic Regression','Random Forest','Support Vector Machine']
modelnames = ['tree.pkl','LogisticRegression.pkl','RandomForest.pkl','SVM.pkl']

# Pass the input to the models and return the prediction
predictions = []
def predict_heart_disease(data):
    for modelname in modelnames:
        model = pickle.load(open(modelname,'rb'))
        prediction = model.predict(data)
        predictions.append(prediction)
    return predictions

# Create a submit button
if st.button("Submit"):
    st.subheader('Results')
    st.markdown('------------------------')

    result = predict_heart_disease(input_data)

    for i in range(len(predictions)):
        st.subheader(algonames[i])
        if result[i][0] == 0:
            st.write("No Heart Disease Detected")
        else:
            st.write("Heart Disease Detected")
        st.markdown('------------------------')

with tab2:
    import plotly.express as px
    data = {'Decision Tree':80.97,'Logistic Regression':85.86,'Random Forest': 84.23,'SVM':84.22}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models,Accuracies)),columns=['Models','Accuracies'])
    fig = px.bar(df, y='Accuracies',x='Models')
    st.plotly_chart(fig)
