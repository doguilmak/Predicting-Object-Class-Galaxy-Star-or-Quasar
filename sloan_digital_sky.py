# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 00:08:09 2021

@author: doguilmak

dataset: https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey

"""
#%%
# 1. Libraries

from keras.models import load_model
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

#%%
#  2. Data Preprocessing

#  2.1. Uploading data
start = time.time()
df = pd.read_csv('Skyserver_SQL2_27_2018 6_51_39 PM.csv')
print(df.head())
print(df.info())

# 2.2. Removing Unnecessary Columns
df.drop(['objid','fiberid'], axis = 1, inplace = True)

# 2.3. Looking for NaN Values
print("Number of NaN values: ", df.isnull().sum().sum())

# 2.4. Looking for Duplicated Rows
print("{} duplicated data.".format(df.duplicated().sum()))

# 2.5. Looking for anomalies
print(df.describe().T)

# 2.6. Label Encoding
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
print("data:\n", df)

# 2.7. Determination of Dependent and Independent Variables
y = df["class"]
X = df.drop("class", axis = 1)

# 2.8.  Splitting Test and Train 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 2.9.  Scaling Datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#%%
# 3 Artificial Neural Network

# 3.1 Loading Created Model
classifier = load_model('model.h5')

# 3.2 Checking the Architecture of the Model
classifier.summary()

"""
# 3.3 Importing libraries
from keras.models import Sequential
from keras.layers import Dense

# 3.4. Creating layers
# Activations link: https://keras.io/api/layers/activations/
# activation="sigmoid"
# activation="relu"
# activation="softmax"
# activation="softplus"

classifier = Sequential()
# Creating first hidden layer:
classifier.add(Dense(32, init="uniform", activation="relu", input_dim=15))
# Creating second hidden layer:
classifier.add(Dense(16, init="uniform", activation="relu"))
# Creating third hidden layer:
classifier.add(Dense(32, init="uniform", activation="relu"))
# Creating output layer:
classifier.add(Dense(3, init="uniform", activation="softmax"))

# Creating output layer:
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
classifier.summary()
#classifier.save('model.h5')
classifier_history = classifier.fit(X, y, epochs=128, batch_size=32, validation_split=0.13)

# 3.4. Plot accuracy and val_accuracy
print(classifier_history.history.keys())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(classifier_history.history['accuracy'])
plt.plot(classifier_history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""

# 3.5. Prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

predict = X.iloc[0:1, ]
if classifier.predict_classes(predict) == 0:
    print('\nModel predicted as GALAXY.')
    print(f'Model predicted class as {classifier.predict_classes(predict)}.')
elif classifier.predict_classes(predict) == 1:
    print('\nModel predicted as QS0.')
    print(f'Model predicted class as {classifier.predict_classes(predict)}.')   
else:
    print('\nModel predicted as STAR')
    print(f'Model predicted class as {classifier.predict_classes(predict)}.')

#%%
# 4 XGBoost

# 4.1 Impotring XGBoost Classifier
from xgboost import XGBClassifier
model=XGBClassifier()

# 4.2. Fit XGBoost
model.fit(X_train, y_train)

# 4.3. Prediction
y_pred_XGBoost = model.predict(X_test)

print('\nXGBoost Prediction')
# 1st column = GALAXY accuracy
# 2nd column = QUASAR accuracy
# 3rd column = STAR accuracy
predict_model_XGBoost = X.iloc[0:1, ]
print(f'Model output: {classifier.predict(predict_model_XGBoost)}.') 

# 4.4. XGBoost Accuracy Score
from sklearn.metrics import accuracy_score
print(f"\nAccuracy score(XGBoost): {accuracy_score(y_test, y_pred_XGBoost)}")

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
