import numpy as np

#Data (Age vs Annual Premium in rupees)
#Age of 5 different people
age =np.array ([18,25,35,45,55])

#their actual premiums
premium= np.array ([15000,10000,21000,24000,29000])

def train_basic_regression(x,y):
    # calculate the means
    x_mean=np.mean(x)
    y_mean=np.mean(y)

    #Calculate slope(m)
    #formula : sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)

    numerator= np.sum((x - x_mean) * (y - y_mean))

    denominator= np.sum((x - x_mean) ** 2)
    m=numerator / denominator

    #calculate intercept(b)
    #formula : y_mean - m * x_mean
    b = y_mean - (m * x_mean)

    return m, b

# 2. train the model
slope , intercept = train_basic_regression(age,premium)

#3.predict
def predict_premium(user_age):
    return (slope * user_age) + intercept

#test
test_age = 25
result = predict_premium(test_age)

print(f"----Healthcare predictor----")
print(f"Cost increase per year (slope): \u20B9{slope:.2f}")

print(f"Base insurance cost (intercept): \u20B9{intercept:.2f}")

print(f"Predicted premium for age {test_age}: \u20B9{result:.2f}")


