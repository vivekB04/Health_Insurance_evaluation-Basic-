import  streamlit as st
import joblib
import pandas as pd

#load the Brain
model = joblib.load("healthcare_model.pkl")

# Set up the page title
st.set_page_config(page_title="Healthcare Premium Predictor")
st.title("Healthcare Premium Predictor")
st.write("Enter your details below to estimate your annual insurance premium.")

#3. create the UI inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider("select your age",18,100,30)

    bmi= st.number_input("Enter your BMI",min_value=10.0,max_value=50.0,value=25.0)

with col2:
    smoker_choice = st.selectbox("Are you a Smoker?",["Yes","No"])

#4 predict Button
if st.button(" Calculate Premium"):
    #Convert inputs into match model format
    smoker = 1 if smoker_choice=="Yes" else 0

    input_df=pd.DataFrame([[age,bmi,smoker]],columns=["age","bmi","smoker"])

    #get prediction
    prediction = model.predict(input_df)[0]

    #display result
    st.success(f"Estimated Annual Premium : \u20B9{prediction:.2f}")

    st.info("Note this ia Basic ML model estimate on provide training data")