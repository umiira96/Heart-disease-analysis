# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:49:18 2022

@author: umium
"""

import pandas as pd
import os
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


DATA_SAVE_PATH = os.path.join(os.getcwd(),'heart.csv')
SCALER_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'r.scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.pkl')


#%%
#EDA 
#Step 1) Data loading
df = pd.read_csv(DATA_SAVE_PATH)

#%%
#Step 2)Data inpection
#shape of data
df.shape

#show summary
con_clms = ["age","trtbps","chol","thalachh","oldpeak"]
df[con_clms].describe().T

#to inspect missing value
df.isnull().sum()
df.boxplot()

#%%
#Step 3)Features selection

# define the columns to be encoded and scaled
cat_clms = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_clms = ["age","trtbps","chol","thalachh","oldpeak"]


# encoding the categorical columns
df = pd.get_dummies(df, columns = cat_clms, drop_first = True)

# define features and target
X = df.drop(['output'],axis=1)
y = df[['output']]

#%%
#Step 4)Data preprocessing
scaler = RobustScaler()

# scaling the continuous features
X[con_clms] = scaler.fit_transform(X[con_clms])
X.head()


pickle.dump(scaler, open(SCALER_SAVE_PATH, 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

#%%
#Step 5)Model creation
model = RandomForestClassifier()

# fitting the model
model.fit(X_train, y_train)

#%% 
#Step 6)Model analysis

y_pred = pd.DataFrame(model.predict(X_test))
y_true = y_test

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

#%% save model
pickle.dump(model, open(MODEL_PATH, 'wb'))














