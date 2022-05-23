# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:39:43 2022

@author: umium
"""

import pickle
import os
import numpy as np
import streamlit as st


#%% PATH
MMS_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'mms.pickle')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_model', 'rf.pkl')

#%% Loading
# mms scaler
with open(MMS_SAVE_PATH, 'rb') as file:
    mms = pickle.load(file)
    
# random forest model
with open(MODEL_SAVE_PATH, 'rb') as file:
    random_forest = pickle.load(file)
    
chances = {0:"low", 1:"high"}

#%% Build app using streamlit 
with st.form('Heart Attack Prediction Form'):
    st.write("Patient's Info")
    age = int(st.number_input("Age")) 
    sex = int(st.number_input("Sex")) 
    cp = int(st.number_input("Chest pain (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic)"))
    trtbps = int(st.number_input("Resting blood pressure (in mm Hg)"))
    chol = int(st.number_input("Cholestoral in mg/dl fetched via BMI se"))
    fbs = int(st.number_input("Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)"))
    restecg = int(st.number_input("Resting electrocardiographic results (0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy)"))
    thalachh = int(st.number_input("Maximum heart rate achieved"))
    oldpeak = st.number_input("Previous peak")
    slp = int(st.number_input("Slope"))
    caa = int(st.number_input("Number of major vessels"))
    thall = int(st.number_input("Thalium Stress Test result ~ (0,3)"))
    exng = int(st.number_input("Exercise induced angina (1 = Yes, 0 = No)"))
    
    
    submitted = st.form_submit_button('Submit')
    
    if submitted == True:
        patience_info = np.array([age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall])
        patience_info = mms.transform(np.expand_dims(patience_info, axis=0))
        predict = random_forest.predict(patience_info)
        if np.argmax(predict) == 1:
            st.success(f"Chances to get heart attack is {chances[np.argmax(predict)]}")
        else:
            st.warning(f"Chances to get heart attack is {chances[np.argmax(predict)]}")