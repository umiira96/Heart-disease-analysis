# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:46:17 2022

@author: umium
"""

import pandas as pd
import os
import pickle 

#path
MMS_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'mms.pickle')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_model', 'rf.pkl')


#%%
#EDA 
#Step 1) Data loading
DATA = os.path.join(os.getcwd(), 'heart.csv')
df = pd.read_csv(DATA)

#%%
#Step 2)Data inpection
#shape of data
df.shape
df.info()
df.describe().T

#to check for duplicate values
df.duplicated().sum() #1 duplicate values

#drop duplicated value
df1 = df.drop_duplicates()
df1.duplicated().sum()

#to check missing value
df.isnull().sum() #no missing values
df.boxplot()

#%%
#Step 3)Features selection
#defining the features and target
X = df1.drop(['output'],axis=1)
y = df1[['output']]

# instantiating the scaler
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
x_scaled = mms.fit_transform(X)
y_scaled = mms.fit_transform(y)

#saved mms
with open(MMS_SAVE_PATH, 'wb') as file:
    pickle.dump(mms, file)

#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled,y_scaled, 
                                                    test_size = 0.2, 
                                                    random_state = 42)

#%%
# Model creation
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# fitting the model
y_train=y_train.ravel()
rf.fit(X_train, y_train)

#%% model analysis
from sklearn.metrics import accuracy_score

y_pred = rf.predict(X_test)
y_true = y_test

print("Test accuracy score of Random Forest is ", accuracy_score(y_pred, y_true))

#%% Model deployment
with open(MODEL_SAVE_PATH, 'wb') as file:
    pickle.dump(rf, file)

