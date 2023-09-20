# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
   1. Import the standard libraries.
   2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
   3. Import LabelEncoder and encode the dataset.
   4.Import LogisticRegression from sklearn and apply the model on the dataset.
   5. Predict the values of array.
   6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
   7. Apply new unknown values.

   

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M.Dinesh
RegisterNumber:  212222040039
*/
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(20)
dataset.tail(20)

#droping tha serial no salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('salary',axis=1)
dataset

dataset = dataset.drop('gender',axis=1)
dataset = dataset.drop('ssc_b',axis=1)
dataset = dataset.drop('hsc_b',axis=1)
dataset

dataset.shape
(215, 10)

dataset.info()

#catgorising for further labeling
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset.dtypes

dataset.info()

dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset
dataset.info()
dataset

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
dataset.head()
x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0]])

*/

 Output:

![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/7739ff31-fef8-4f34-8c87-585a57a88192)

![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/aad8cdc4-a792-4849-83fd-cbe34de30b25)


![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/7adf4de4-ae75-4b47-90c5-b28495b39fa8)

![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/17fc7934-c1cb-43d0-86da-368cea0eea7f)


![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/40b87c44-cf69-4657-b075-99e0c85fa0eb)

![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/fffbed93-8031-4f67-91ac-9ec19560663f)

![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/cb305b0b-2ec1-4d36-8641-c93fec2f969a)

![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/1f0f866c-d427-4e84-916b-7efe7ea7e2a2)

![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/aa254c36-3140-462f-87b4-0bf12e8db9e0)

![image](https://github.com/dineshmohan24102004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119478475/1d9052be-3a5a-4ce6-96d4-941314a7bbbe)












 Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
