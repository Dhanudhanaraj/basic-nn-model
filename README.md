# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 2 hidden layer(1st hidden layer contains 7 neurons and 2nd hidden layer contains 14 neurons) with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/d3f2c4e7-350e-41fd-b57c-5daae9200396)

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
df.head()

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
    Dense(7,activation='relu'),
    Dense(14,activation='relu'),
    Dense(1)
])

#Compile the model
ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')

# Fit the model
ai_brain.fit(X_train1 , y_train,epochs = 3000)

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()
X_test1 =Scaler.transform(x_test)
ai_brain.evaluate(X_test1,y_test)
X_n1=[[11]]
X_n1_1=Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information
![Screenshot 2024-02-20 075748](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/a1e7940a-5fdc-4605-adfd-5a2373647ee8)

## OUTPUT

### Training Loss Vs Iteration Plot
![Screenshot 2024-02-20 082330](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/b3069572-142a-4121-b5d2-48d63a534a81)

### Test Data Root Mean Squared Error
![Screenshot 2024-02-20 082431](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/f3eb9b29-06c3-4d8a-91b3-f6c1f77b0baa)

![Screenshot 2024-02-20 082446](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/6ed384bd-65e4-4f12-8946-eae45a0e62d4)

### New Sample Data Prediction
![Screenshot 2024-02-20 082314](https://github.com/Dhanudhanaraj/basic-nn-model/assets/119218812/6b089717-9864-4481-a9b0-b8cd620d834c)

## RESULT
Thus a neural network regression model for the given dataset is written and executed successfully
