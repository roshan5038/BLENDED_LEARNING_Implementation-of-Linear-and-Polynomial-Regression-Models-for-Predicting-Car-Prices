# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('encoded_car_data (1).csv')
print(df.head())

X = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

lr = Pipeline([('scaler' , StandardScaler()), ('model' , LinearRegression())])
lr.fit(X_train,y_train)
y_pred_linear = lr.predict(X_test)

poly_model = Pipeline([
    ('poly' , PolynomialFeatures(degree = 2)),
    ('scaler' , StandardScaler()),
    ('model', LinearRegression())
])
poly_model.fit(X_train,y_train)
y_pred_poly = poly_model.predict(X_test)

print("Name: ")
print("Reg No: ")
print("Linear Regression: ")
print('MSE=',mean_squared_error(y_test,y_pred_linear))
r2score = r2_score(y_test,y_pred_linear)
print("R2 Score= ", r2score)

print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test,y_pred_poly):.2f}")
print(f"R2: {r2_score(y_test , y_pred_poly):.2f}")

plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred_linear , label = 'Linear' , alpha = 0.6)
plt.scatter(y_test , y_pred_poly , label = 'Polynomial (degree =2)' , alpha=0.6)
plt.plot([y.min() , y.max()] , [y.min(),y.max()], 'r--' , label ='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
