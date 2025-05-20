# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results. 

## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: DINESH PRABHU S

RegisterNumber: 212224040077
*/
```
import pandas as pd
df=pd.read_csv("placement_Data.csv")
df.head()

data1=df.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

new_data = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
placement_status = model.predict(new_data)
print("Predicted Placement Status:", placement_status)
```

## Output:
![image](https://github.com/user-attachments/assets/439ea739-c13a-47c4-84a1-f07074d20ca0)

![image](https://github.com/user-attachments/assets/fa27b438-3832-4599-8dff-fbaaf3a6c842)

![image](https://github.com/user-attachments/assets/91efc5f8-562f-4dae-91cf-da1c07ef5270)

![image](https://github.com/user-attachments/assets/03da175f-5983-494b-a6fc-fa2b4db704d0)

![image](https://github.com/user-attachments/assets/d49d93f2-5204-407a-aced-824497c4b19c)

![image](https://github.com/user-attachments/assets/0e196b4a-458d-40c4-8105-d300f2f6e3f1)

![image](https://github.com/user-attachments/assets/401e0c36-022e-4738-8fe5-6c50d6b634ff)

![image](https://github.com/user-attachments/assets/1b89d43e-2386-482c-8e4e-ec31f1fdf388)

![image](https://github.com/user-attachments/assets/55d39fde-6b45-43a6-8d23-a3fa914f0069)

![image](https://github.com/user-attachments/assets/14f77481-1309-461f-808f-df4349e23649)

![image](https://github.com/user-attachments/assets/6a98442c-6d01-46c9-921d-1e24f3035917)

![image](https://github.com/user-attachments/assets/0eb3966c-ad79-4fd3-bcb8-5adb8b44a827)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
