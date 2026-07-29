import numpy as np
import pandas as pd
import os

df=pd.read_csv('Churn_Modelling.csv')
#print(df.head())
#print(df.shape)
#df.info()
#print(df['Exited'].value_counts())
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

from sklearn.model_selection import train_test_split

X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential

