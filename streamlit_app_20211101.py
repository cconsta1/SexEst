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

# Loading the Howells models

logreg_model_howell = pickle.load(open("logreg_model_howell.dat", "rb"))
svm_model_howell = pickle.load(open("svm_model_howell.dat", "rb"))
lda_model_howell = pickle.load(open("lda_model_howell.dat", "rb"))

pressed = False

if st.sidebar.button("About"):
    pressed = not pressed

    with st.container():
           st.write("This is inside the container")


with st.sidebar.form(key='test'):
    if st.form_submit_button("Hi"):
        st.write("Hello")

with st.sidebar.expander("Osteometric Sex Prediction (single entry)"):
    BIB = st.number_input("BIB", min_value=0.0,
                          max_value=500.0, value=0.0, step=0.1)
    HML = st.number_input("HML")
    HHD = st.number_input("HHD")
    RML = st.number_input("RML")
    FML = st.number_input("FML")
    FBL = st.number_input("FBL")
    FHD = st.number_input("FHD")
    TML = st.number_input("TML")

    x = [[BIB, HML, HHD, RML, FML, FBL, FHD, TML]]

    x = np.asarray(x, dtype='float64')

    is_all_zero_osteometric = not x.all()

    model_selection = st.selectbox(
        "Select a model ", options=["XGB", "LGB", "ADA"])

    st.button("Press me")
    

with st.sidebar.expander("Craniometric Sex Prediction (single entry)"):
    GOL = st.number_input("GOL")
    NOL = st.number_input("NOL")
    BNL = st.number_input("BNL")
    BBH = st.number_input("BBH")
    XCB = st.number_input("XCB")
    XFB = st.number_input("XFB")
    ZYB = st.number_input("ZYB")
    AUB = st.number_input("AUB")
    WCB = st.number_input("WCB")
    ASB = st.number_input("ASB")
    BPL = st.number_input("BPL")
    NPH = st.number_input("NPH")
    NLH = st.number_input("NLH")
    JUB = st.number_input("JUB")
    NLB = st.number_input("NLB")
    MAB = st.number_input("MAB")
    MDH = st.number_input("MDH")
    MDB = st.number_input("MDB")
    OBH = st.number_input("OBH")
    OBB = st.number_input("OBB")
    DKB = st.number_input("DKB")
    ZMB = st.number_input("ZMB")
    FMB = st.number_input("FMB")
    EKB = st.number_input("EKB")
    IML = st.number_input("IML")
    XML = st.number_input("XML")
    WMH = st.number_input("WMH")
    STB = st.number_input("STB")
    FRC = st.number_input("FRC")
    PAC = st.number_input("PAC")
    OCC = st.number_input("OCC")
    FOL = st.number_input("FOL")

    y = [[GOL, NOL, BNL, BBH, XCB, XFB, ZYB, AUB, WCB, ASB, BPL, NPH, NLH, JUB, NLB, MAB,
          MDH, MDB, OBH, OBB, DKB, ZMB, FMB, EKB, IML, XML, WMH, STB, FRC, PAC, OCC, FOL]]

    y = np.asarray(y, dtype='float64')

    is_all_zero_craniometric = not y.all()

    model_selection_howell = st.selectbox(
        "Select a model ", options=["LOGREG", "SVM", "LDA"])

    st.write(is_all_zero_craniometric)

with st.sidebar.expander("Osteometric Sex Prediction (file entry)"):

    df = pd.DataFrame()

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None,
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7])

    model_selection = st.selectbox(
        "Select a model", options=["XGB", "LGB", "ADA"])

with st.sidebar.expander("Craniometric Sex Prediction (file entry)"):

    df_howell = pd.DataFrame()

    uploaded_file = st.file_uploader("Choose a CSV file ", type=['csv'])

    if uploaded_file is not None:
        df_howell = pd.read_csv(uploaded_file, header=None, usecols=[
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

    st.dataframe(df_howell)

    model_selection_howell = st.selectbox(
        "Select a model", options=["LOGREG", "SVM", "LDA"])


with st.sidebar.expander("Contact"):
    st.write("Contact")


def main():
    st.title("SexEst: A sex estimation web application")
    html_temp = """
    <div style="background-color:tomato;padding:8px">
    <h3 style="color:white;text-align:center;"> Results section </h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    placeholder = st.empty()

    if not is_all_zero_osteometric:
        placeholder.empty()

        placeholder.write(x)

        result = predict_sex_osteometric(x, model_selection)

        placeholder.success('The osteometric sex is {}'.format(result))

    if (df.empty != True) and (model_selection != ""):
        predict_sex_with_file_osteometric(df, model_selection)

    if not is_all_zero_craniometric:
        placeholder.empty()

        placeholder.write(y)

        result = predict_sex_craniometric(y, model_selection_howell)

        placeholder.success('The craniometric sex is {}'.format(result))

    if (df_howell.empty != True) and (model_selection_howell != ""):
        predict_sex_with_file_craniometric(df_howell, model_selection_howell)

    if (pressed):
        st.empty()
        st.write("Geia sou malaka")
    else:
        st.write("Bastard")


def predict_sex_osteometric(x, model):

    if (model == "XGB"):
        predict_proba = xgb_model_goldman.predict_proba(x)

        return predict_proba

    if (model == "LGB"):
        predict_proba = lgb_model_goldman.predict_proba(x)

        return predict_proba

    if (model == "ADA"):
        predict_proba = ada_boost_model_goldman.predict_proba(x)

        return predict_proba


def predict_sex_craniometric(x, model):

    if (model == "LOGREG"):
        predict_proba = logreg_model_howell.predict_proba(x)

        return predict_proba

    if (model == "SVM"):
        predict_proba = svm_model_howell.predict_proba(x)

        return predict_proba

    if (model == "LDA"):
        predict_proba = lda_model_howell.predict_proba(x)

        return predict_proba


def predict_sex_with_file_osteometric(df, model):

    df = df.dropna()
    predict_proba = []

    if (model == "XGB"):
        predict_proba = xgb_model_goldman.predict_proba(df.values)

    if (model == "LGB"):
        predict_proba = lgb_model_goldman.predict_proba(df.values)

    if (model == "ADA"):
        predict_proba = ada_boost_model_goldman.predict_proba(df.values)

    st.header('Your input')
    st.dataframe(df)
    st.header('The output')
    st.write(predict_proba)


def predict_sex_with_file_craniometric(df, model):

    df = df.dropna()
    predict_proba = []

    if (model == "LOGREG"):
        predict_proba = logreg_model_howell.predict_proba(df.values)

    if (model == "SVM"):
        predict_proba = svm_model_howell.predict_proba(df.values)

    if (model == "LDA"):
        predict_proba = lda_model_howell.predict_proba(df.values)

    st.header('Your input')
    st.dataframe(df)
    st.header('The output')
    st.write(predict_proba)


if __name__ == '__main__':
    main()
