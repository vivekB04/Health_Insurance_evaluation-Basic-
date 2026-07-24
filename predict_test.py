import joblib
import pandas as pd

#1 Load the saved model

model= joblib.load('healthcare_model.pkl')

#2. Create a function that the Frontend will eventually call

def get_prediction(age, bmi, smoker_text):
    # Convert "Yes/No" text from UI to 1/6 for the model
    smoker = 1 if smoker_text.lower() == 'yes' else 0

    # Wrap input in a DataFrame so the model recognizes feature names
    input_data = pd.DataFrame([[age, bmi, smoker]],
                              columns=['age', 'bmi', 'smoker'])
    prediction = model.predict(input_data)
    return prediction[0]

#3. Test it manually

result = get_prediction(age=30, bmi=25.0,smoker_text='no')

print(f"Step 2 Success: The loaded model predicted a premium of {result:.2f}")