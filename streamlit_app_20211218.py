"""
 @file streamlit_app.py
 A web app for deploying sex estimation machine learning
 models
 
 Language: python
 
 Chrysovalantis Constantinou
 
 The Cyprus Institute
 
 + 10/28/21 (cc): Created.
 
"""

from datetime import time
import numpy as np
import pickle
import pandas as pd
import streamlit as st 

# Loading the Goldman models 

ada_boost_model_goldman = pickle.load(open("ada_boost_model.dat", "rb"))
xgb_model_goldman = pickle.load(open("xgb_model.dat", "rb"))
lgb_model_goldman = pickle.load(open("lgb_model.dat", "rb"))




# def welcome():
#     return "Welcome All"


def predict_sex_osteometric_xgb_classifier(BIB,HML,HHD,RML,FML,FBL,FHD,TML):
   
    x = [[BIB,HML,HHD,RML,FML,FBL,FHD,TML]]

    x = np.asarray(x, dtype='float64')

    #prediction = xgb_model_goldman.predict(x)

    predict_proba = xgb_model_goldman.predict_proba(x)

    return predict_proba




def predict_sex_osteometric_lgb_classifier(BIB,HML,HHD,RML,FML,FBL,FHD,TML):
    
    x = [[BIB,HML,HHD,RML,FML,FBL,FHD,TML]]

    x = np.asarray(x, dtype='float64')

    #prediction=lgb_model_goldman.predict(x)

    predict_proba = lgb_model_goldman.predict_proba(x)

    return predict_proba

def predict_sex_osteometric_ada_boost_classifier(BIB,HML,HHD,RML,FML,FBL,FHD,TML):
  x = [[BIB,HML,HHD,RML,FML,FBL,FHD,TML]]

  x = np.asarray(x, dtype='float64')

  #prediction=lgb_model_goldman.predict(x)

  predict_proba = ada_boost_model_goldman.predict_proba(x)

  return predict_proba

def predict_sex_with_file_osteometric_xgb_classifier(df):

  df=df.dropna()  

  prediction=xgb_model_goldman.predict_proba(df.values)

  #prediction = pd.DataFrame(prediction)
  st.header('Your input')
  st.dataframe(df)
  st.header('The output')
  st.write(prediction)

  #return prediction

def get_params():
  BIB=st.text_input("BIB", "")
  HML=st.text_input("HML", "")
  HHD=st.text_input("HHD", "")
  RML=st.text_input("RML", "")
  FML=st.text_input("FML", "")
  FBL=st.text_input("FBL", "")
  FHD=st.text_input("FHD", "")
  TML=st.text_input("TML", "")

  x = [[BIB,HML,HHD,RML,FML,FBL,FHD,TML]]

  x = np.asarray(x, dtype='float64')

  return x


def main():
    st.title("SexEst: A sex estimation web application")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Awesome app... </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


    st.sidebar.button("Predict Sex Using XGB boosting")

      # BIB=st.text_input("BIB", "")
      # HML=st.text_input("HML", "")
      # HHD=st.text_input("HHD", "")
      # RML=st.text_input("RML", "")
      # FML=st.text_input("FML", "")
      # FBL=st.text_input("FBL", "")
      # FHD=st.text_input("FHD", "")
      # TML=st.text_input("TML", "")
      # result = ""

    x = get_params()

    st.write(x)


      # result=predict_sex_osteometric_xgb_classifier(BIB, HML, HHD, RML, FML, FBL, FHD, TML)
      # st.success('The sex is {}'.format(result))



    # BIB=st.text_input("BIB", "")
    # HML=st.text_input("HML", "")
    # HHD=st.text_input("HHD", "")
    # RML=st.text_input("RML", "")
    # FML=st.text_input("FML", "")
    # FBL=st.text_input("FBL", "")
    # FHD=st.text_input("FHD", "")
    # TML=st.text_input("TML", "")
    # result=""

    # if st.button("Predict Sex Using XGB boosting"):
    #   result=predict_sex_osteometric_xgb_classifier(BIB, HML, HHD, RML, FML, FBL, FHD, TML)
    #   st.success('The sex is {}'.format(result))

    if st.sidebar.button("Predict Sex Using LGB boosting"):
      result=predict_sex_osteometric_lgb_classifier(BIB, HML, HHD, RML, FML, FBL, FHD, TML)
      st.success('The sex is {}'.format(result))

    if st.sidebar.button("Predict Sex Using ADA boosting"):
      result=predict_sex_osteometric_ada_boost_classifier(BIB, HML, HHD, RML, FML, FBL, FHD, TML)
      st.success('The sex is {}'.format(result))

    

    # st.header("Predict gender using your CSV file")
    # uploaded_file = st.file_uploader("Choose a file", type = ['csv'])
  
    # if uploaded_file is not None:
        
    #   df=pd.read_csv(uploaded_file, header=None, usecols=[0,1,2,3,4,5,6,7])
      

    #   predict_sex_with_file_osteometric_xgb_classifier(df)
        


    # if st.button("About"):
    #     st.text("About section")
    #     st.text("...")

if __name__=='__main__':
    main()
    
