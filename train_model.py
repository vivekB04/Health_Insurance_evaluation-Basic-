import pandas as pd

from sklearn.linear_model import LinearRegression

import joblib #This Library saves the model to your hard drive

#1 Dataset (Minimal Option A)

data ={ 'age': [15, 22, 35, 45, 50, 19, 33, 60, 25, 40],
        'bmi': [22.5, 28.1, 25.0, 32.2, 29.5, 20.1, 38.0, 27.0, 24.0, 30.0],
        'smoker': [0, 0, 1, 0, 1, 0, 1, 0, 0, 1], #0: No, 1: Yes
        'premium': [12000, 14000, 25000, 22000, 35000, 11000, 32000, 28000, 15000, 29000]
        }

df=pd.DataFrame (data)

#2. Split Features and Target

x=df[['age', 'bmi', 'smoker']]
y= df['premium']

#3. Train the Model

model = LinearRegression()
model.fit(x, y)

#4. SAVE THE MODEL
#This creates a file in your project folder

joblib. dump (model, 'healthcare_model.pkl')

print("Step I Success: Model trained and saved as healthcare model.pkl")