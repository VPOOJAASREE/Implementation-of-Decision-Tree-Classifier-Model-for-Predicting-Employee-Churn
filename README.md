# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: V. POOJAA SREE
RegisterNumber: 212223040147  
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

DATA:

![o1](https://github.com/user-attachments/assets/e07e6f2f-bfa1-4b04-82fe-6271a99a51c4)

![o2](https://github.com/user-attachments/assets/c8a4d946-da2b-4667-8014-1a952f904291)

![o3](https://github.com/user-attachments/assets/0e2dfd83-1828-4f2f-b4f4-a2f9ad0bca62)

ACCURACY:

![o4](https://github.com/user-attachments/assets/1d6e6bd4-054b-4963-90b8-13309c74454b)

PREDICT:

![o5](https://github.com/user-attachments/assets/7156e1a8-93d6-4fb7-b0d6-2ac205e7171e)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
