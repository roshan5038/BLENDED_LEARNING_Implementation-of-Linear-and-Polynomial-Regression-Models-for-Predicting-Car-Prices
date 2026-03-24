# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the car price dataset.

2.Select input features and target variable (price).

3.Split the dataset into training and testing data.

4.Train Linear Regression and Polynomial Regression models.

5.Predict prices, evaluate performance using metrics, and compare the results.

## Program:
```
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: ROSHAN V
RegisterNumber: 25004228   //   212225240124
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

df = pd.read_csv("encoded_car_data.csv")
print(df.head())

X = df[['horsepower','enginesize', 'citympg', 'highwaympg']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr=Pipeline([('scaler',StandardScaler()),
            ('model', LinearRegression())])
lr.fit(X_train, y_train)
y_pred_linear=lr.predict(X_test)

poly_model = Pipeline([('poly',PolynomialFeatures(degree=2)),
                      ('scaler',StandardScaler()),
                      ('model',LinearRegression())])
poly_model.fit(X_train,y_train)
y_pred_poly=poly_model.predict(X_test)


print("Name: ROSHAN V ")
print("Reg No: 212225240124 ")
print("Linear Regression:")

print(f"{'MSE'}: {mean_squared_error(y_test,y_pred_linear)}")
r2score=r2_score(y_test,y_pred_linear)
print("R2 Score=",r2score)

print("\nPolynomial Regression:")
print(f"{'MSE'}: {mean_squared_error(y_test,y_pred_poly)}")

print(f"{'R-squared'}: {r2_score(y_test,y_pred_poly)}")

plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred_linear,label='Linear',alpha=0.6)
plt.scatter(y_test,y_pred_poly,label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--',label='Perfect Prediction')
plt.title("Linear vs Polynomial Prediction")
plt.xlabel("Actual Price ")
plt.ylabel("Predicted Price ")
plt.legend()
plt.show()
```

## Output:
<img width="1920" height="1080" alt="Screenshot 2026-02-11 091518" src="https://github.com/user-attachments/assets/a805fc61-3937-4630-ba51-e0712c931732" />

<img width="1920" height="1080" alt="Screenshot 2026-02-11 091541" src="https://github.com/user-attachments/assets/9f1db24c-ab9a-418e-9028-210bd8b5ae0a" />



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
