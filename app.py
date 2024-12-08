# IMPORTING THE DEPENDENCIES

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# DATA COLLECTION AND ANALYSIS

# Loading the dataset
data = pd.read_csv("diabetes.csv")

# Printing 1st 5 rows
print(data.head())

# Number of rows and columns
print(data.shape)

# Getting the statistical measures of the data
print(data.describe())

# 0 --> Non-Diabetic
# 1 --> Diabetic
print(data['Outcome'].value_counts())

print(data.groupby('Outcome').mean())

# Separating the data and labels
X = data.drop(columns='Outcome',axis = 1)
Y = data['Outcome']

print(X)

print(Y)

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized = scaler.transform(X)
print(standardized)

X = standardized
Y = data['Outcome']

print(X)
print(Y)

#TRAIN TEST SPLIT
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)

# MODEL TRAINING

classifier = svm.SVC(kernel='linear')

#Training the support vector Machine Classifier
print(classifier.fit(X_train, Y_train))

# MODEL EVALUATION

# Accuracy Score on Training Data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy Score on Training Data : ', training_data_accuracy)

# Accuracy Score on Test Data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy Score on Test Data : ', test_data_accuracy)

# MODEL PREDICTION

input_data = (0,137,40,35,168,43.1,2.288,33)

# Changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic') 




