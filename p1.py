# import lib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
# load the data

data = pd.read_csv("heart_sett.csv")
print(data.head())

# understand the data
print(data.isnull().sum())

# feature engineering
features = data[["chol","trestbps","thalach","fbs"]]
target = data["target"]

# handling the data
new_features = pd.get_dummies(features,drop_first=True)

# train and test
x_train,x_test,y_train,y_test = train_test_split(new_features,target,random_state=123)

# model and fit
model = RandomForestClassifier()
model.fit(x_train,y_train)

# classification report
y_pred = model.predict(x_test)
cr = classification_report(y_test,y_pred)
print(cr)

# save the model
with open("db.model","wb") as f:  
	pickle.dump(model,f)