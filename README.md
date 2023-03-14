# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![](img/0.svg)

## DESIGN STEPS

#### STEP 1:

Loading the dataset

#### STEP 2:

Split the dataset into training and testing

#### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

#### STEP 4:

Build the Neural Network Model and compile the model.

#### STEP 5:

Train the model with the training data.

#### STEP 6:

Plot the performance plot

#### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

~~~python
### To Read CSV file from Google Drive :
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

### Authenticate User:
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :
worksheet = gc.open('Dataset 1.0').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

### Import the packages :
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=df[['Input']].values
y=df[['Output']].values

x
y

### Split Training and testing set :
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)

### Pre-processing the data :
scaler=MinMaxScaler()
scaler.fit(xtrain)

xtrain1=scaler.transform(xtrain)

### Model :
model_1_0=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])

### Loss plot :
model_1_0.compile(optimizer='rmsprop',loss='mse')
model_1_0.fit(xtrain1,ytrain,epochs=5000)
loss=pd.DataFrame(model_1_0.history.history)
loss.plot()

### Testing with the test data and predicting the output :
xtest1=scaler.transform(xtest)
model_1_0.evaluate(xtest1,ytest)
xn1=[[4]]
xn1_1=scaler.transform(xn1)
model_1_0.predict(xn1_1)
~~~

### Dataset Information

![](img/1.JPG)

## OUTPUT

#### Training Loss Vs Iteration Plot

![](img/2.JPG)

#### Test Data Root Mean Squared Error

![](img/3.JPG)

#### New Sample Data Prediction

![](img/4.JPG)

## RESULT
