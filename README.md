# <p align="center">Developing a Neural Network Regression Model</p>

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network two hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

<p align="center">
    <img alt="image" src="https://user-images.githubusercontent.com/94154252/226111563-e2e31dcd-08f7-4c64-9af7-0a0004fe8b07.JPG">
</p>

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar object, fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```py
# Name            : Venkatesh E
# Register Number : 212221230119
```
```py
### Importing Modules
from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

### Authenticate &  Create Dataframe using Data in Sheets
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('Dataset 1.0').sheet1 
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

### Assign X and Y values
x=df[['Input']].values
y=df[['Output']].values

x
y

### Normalize the values & Split the 
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 1)

scaler=MinMaxScaler()
scaler.fit(xtrain)

xtrain1=scaler.transform(xtrain)

### Create a Neural Network & Train it
model_1_0=Sequential([
    Dense(8,activation='relu'),
    Dense(15,activation='relu'),
    Dense(1)
])

ai.compile(optimizer = 'rmsprop',loss = 'mse')

model_1_0.fit(xtrain1,ytrain,epochs=3000)

### Plot the Loss
loss=pd.DataFrame(model_1_0.history.history)
loss.plot()

### Evaluate the model
xtest1=scaler.transform(xtest)
model_1_0.evaluate(xtest1,ytest)

### Predict for some value
xn1=[[4]]
xn1_1=scaler.transform(xn1)
model_1_0.predict(xn1_1)
```
## Dataset Information

<p align="center">
    <img alt="image" src="https://user-images.githubusercontent.com/94154252/226111684-4f168eae-28a7-47cb-bf6e-add24d738e74.JPG">
</p>


## OUTPUT

### Training Loss Vs Iteration Plot

<p align="center">
    <img alt="image" src="https://user-images.githubusercontent.com/94154252/226111753-4e7ace33-735d-434e-9c31-b13fb9cbcc99.JPG">
</p>

### Test Data Root Mean Squared Error

<p align="center">
    <img alt="image" src="https://user-images.githubusercontent.com/94154252/226111778-e417f8cd-6cf9-4713-adf7-6eb73b906810.JPG">
</p>

### New Sample Data Prediction

<p align="center">
    <img alt="image" src="https://user-images.githubusercontent.com/94154252/226111794-4a9a98b4-d8b8-456b-bdbc-be4ea78abae8.JPG">
</p>


## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
