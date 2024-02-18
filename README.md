# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/e8e1b995-aeca-471e-8696-0122fa0f799c)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
## NAME:DHANUMALYA.D
## REGISTER NUMBER:212222230030

### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

## To train and test
from sklearn.model_selection import train_test_split

## To scale
from sklearn.preprocessing import MinMaxScaler

## To create a neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('Deep Learning').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])

df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})

df

x=df[['Input']].values
y=df[['Output']].values

x

y

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.4, random_state =35)

Scaler = MinMaxScaler()
Scaler.fit(x_train)

X_train1 = Scaler.transform(x_train)

#Create the model
ai_brain = Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])

#Compile the model
ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')

# Fit the model
ai_brain.fit(X_train1 , y_train,epochs = 2005)

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()

X_test1 =Scaler.transform(x_test)

ai_brain.evaluate(X_test1,y_test)

X_n1=[[4]]

X_n1_1=Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)
```
## Dataset Information
![Screenshot 2024-02-18 211218](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/f8fc3950-7b2b-4e49-9a80-1542bdd676ff)

## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2024-02-18 210448](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/b92a7eb6-dc6a-4403-99ce-cc75e1a5201b)


### Test Data Root Mean Squared Error
![Screenshot 2024-02-18 210438](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/6ff22cc2-a8c0-4752-8a1d-bbe0667cbff1)

![Screenshot 2024-02-18 210539](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/7d6daed1-49b5-4382-8fc0-35c96f3f3cb5)


### New Sample Data Prediction
![Screenshot 2024-02-18 210556](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/5ae29b1b-ef3b-4d1f-b41e-30e5e2a0fc01)



## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully
