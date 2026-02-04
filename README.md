# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and prepare the multivariate input data and target values.
2.Initialize the SGD Regressor with appropriate learning rate and iterations.

3.Train the model using the given dataset and predict the output values.

4.Compare actual and predicted values using graphical visualization.

 

## Program:
~~~
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: RAHUL
RegisterNumber:  25003095
*/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
data = fetch_california_housing()
X = data.data[:, :3]
y_price = data.target         
y_occup = data.data[:, 5]     
Y = np.column_stack((y_price, y_occup))
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model = MultiOutputRegressor(sgd)

model.fit(X_train_scaled, Y_train_scaled)
mse_price = mean_squared_error(Y_test_original[:, 0], Y_pred[:, 0])
mse_occup = mean_squared_error(Y_test_original[:, 1], Y_pred[:, 1])

print("Mean Squared Error (House Price):", mse_price)
print("Mean Squared Error (Average Occupants):", mse_occup)

print("\nSample Predictions (Price, Occupants):")
for i in range(5):
    print("Predicted:", Y_pred[i], " | Actual:", Y_test_original[i])
~~~


## Output:

<img width="735" height="219" alt="544113556-370a3752-9397-4b63-82f5-c509f637dd5a" src="https://github.com/user-attachments/assets/5bceeeb0-ba92-4182-89fb-b6b5b4ead535" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
