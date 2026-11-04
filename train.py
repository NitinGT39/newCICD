import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(data = X, columns = iris.feature_names)
df['target'] = y

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lg = LogisticRegression()
lg.fit(X_train, y_train)

y_pred = lg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))



import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump({
        'model': lg,
        'scaler': scaler
    }, file)

print("Model and scaler saved successfully!")


