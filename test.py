import pandas as pd
import numpy as np
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

with open('model.pkl', 'rb') as file:
    artifacts = pickle.load(file)

model = artifacts['model']
scaler = artifacts['scaler']


X_test = scaler.transform(X_test)

predictions = model.predict(X_test)

print("Loaded Model Accuracy:",
acc=accuracy_score(y_test, predictions))

print("Loaded Model Classification Report:")
print(classification_report(y_test, predictions))

assert acc>=90

