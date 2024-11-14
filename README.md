# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import Libraries: Import the necessary libraries such as pandas, numpy, matplotlib, and sklearn.

Load Data: Read the CSV file containing the dataset and display its contents.

Prepare Data: Segregate the data into independent (X) and dependent (Y) variables.

Split Data: Split the dataset into training and testing sets using train_test_split.

Train Model: Import the Linear Regression model from sklearn, fit the model with the training data, and make predictions on the test data.

Evaluate Model: Display the predicted and actual values, plot the training and test data, and calculate the mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE)
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VINOTH M P
RegisterNumber:212223240182  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error,mean_squared_error
#read csv file
df=pd.read_csv('student_scores (2).csv')
#displaying the content in datafile
df.head()

# Segregating data to variables
x= df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,reg.predict(x_train))
plt.title("hours vs marks")
plt.ylabel("marks obtained")
plt.xlabel("hours studied")

#graph plot for test data
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,reg.predict(x_test))
plt.title("hours vs marks")
plt.ylabel("marks obtained")
plt.xlabel("hours studied")

#find mae,mse,rmse
mae=mean_absolute_error(y_test,y_pred)
mae
mse=mean_squared_error(y_test,y_pred)
mse
rmse=np.sqrt(mse)
rmse
```

## Output:

### HEAD VALUES
![Screenshot 2024-09-04 221358](https://github.com/user-attachments/assets/fbda0ea9-ed7d-467d-a43b-8f331d462023)
### X VALUES
![Screenshot 2024-09-04 221417](https://github.com/user-attachments/assets/a9e9dbd6-635a-48e3-8071-cd2a81c55147)
### Y_PRED AND Y_TEST VALUES
![Screenshot 2024-09-04 221437](https://github.com/user-attachments/assets/5676a893-dcaf-4457-aefa-3ab8dbe9e199)
### TRAINING DATA GRAPH
![Screenshot 2024-09-04 221450](https://github.com/user-attachments/assets/0fd7830c-e389-45d0-972f-de148785960b)
### TESTING DATA GRAPH
![Screenshot 2024-09-04 221500](https://github.com/user-attachments/assets/b601a081-16fa-4378-8895-615ab4077b0a)
### ERROR VALUES
![Screenshot 2024-09-04 221509](https://github.com/user-attachments/assets/ed4a97c9-83ca-4f71-b519-9e40b426f98b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
