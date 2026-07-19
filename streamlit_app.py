"""
 @file streamlit_app.py
 A web app for deploying sex estimation machine learning
 models

 Language: python

 Chrysovalantis Constantinou

 The Cyprus Institute

 + 10/28/21 (cc): Created.
 + 11/16/21 (cc): Basic functional version completed
 + 11/23/21 (cc): Added a configuration file to beautify the app
 + 12/18/21 (cc): Added missing data tabs and functionality
 + 03/29/22 (cc): Added Google Analytics functionality
 + 05/30/22 (cc): Added 3 independent variables to the postcranial models
 + 07/18/26 (cc): Fixed malformed CSS and global style leaks causing
   sidebar labels to vanish; reworked navigation to use st.session_state
   so results and screens persist correctly across reruns; cached model
   loading; refreshed dependencies and contact info; general polish.
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
import toml
import os


# Web app wide configuration

st.set_page_config(
    page_title="SexEst",
    page_icon="skull",
    layout="wide"
)

# Fix the button's sizes by passing CSS rules to the web app

primaryColor = toml.load(".streamlit/config.toml")["theme"]["primaryColor"]

sidebar_configuration = """
<style>
section[data-testid="stSidebar"]
    {
        width: 380px !important;
    }
</style>
"""

st.markdown(sidebar_configuration, unsafe_allow_html=True)

button_configuration = f"""
<style>
div.stButton > button:first-child
    {{
        justify-content: justify;
        font-size:13px;
        width: 378px;
        background: transparent;
        height:45.5px;
        box-sizing: border-box;
        width: 100%;
    }}
</style>
"""

st.markdown(button_configuration, unsafe_allow_html=True)

expander_configuration = f"""
<style>
.streamlit-expanderHeader
    {{
        width: 380px;
        justify-content: justify;
        font-size:13px;
    }}
</style>
"""

st.markdown(expander_configuration, unsafe_allow_html=True)

download_button_configuration = f"""
<style>
div.stDownloadButton > button:first-child
    {{
        border: 1px solid {primaryColor};
        border-radius: 4px 4px 4px 4px;
        width: 180px;
        margin: 20px
    }}
</style>
"""

st.markdown(download_button_configuration, unsafe_allow_html=True)

browse_button_configuration = f"""
<style>
div[data-testid="stFileUploader"] section button
    {{
        border: 1px solid {primaryColor};
        border-radius: 4px 4px 4px 4px;
        width: 180px;
    }}
</style>
"""

st.markdown(browse_button_configuration, unsafe_allow_html=True)

# Loading the Goldman models and their accuracy

has_cache_resource = hasattr(st, "cache_resource")


def _cache_models(func):
    if has_cache_resource:
        return st.cache_resource(func)
    return st.cache(allow_output_mutation=True)(func)


@_cache_models
def load_goldman_models():
    xgb_model = pickle.load(open("./models_goldman/model_xgb_goldman.dat", "rb"))
    accuracy_file_xgb = open("./models_goldman/model_xgb_goldman.txt", "r")
    accuracy_xgb = float(accuracy_file_xgb.read()) * 100

    lgb_model = pickle.load(open("./models_goldman/model_lgb_goldman.dat", "rb"))
    accuracy_file_lgb = open("./models_goldman/model_lgb_goldman.txt", "r")
    accuracy_lgb = float(accuracy_file_lgb.read()) * 100

    lda_model = pickle.load(open("./models_goldman/model_lda_goldman.dat", "rb"))
    accuracy_file_lda = open("./models_goldman/model_lda_goldman.txt", "r")
    accuracy_lda = float(accuracy_file_lda.read()) * 100

    return xgb_model, accuracy_xgb, lgb_model, accuracy_lgb, lda_model, accuracy_lda


@_cache_models
def load_howell_models():
    xgb_model = pickle.load(open("./models_howell/model_xgb_howell.dat", "rb"))
    accuracy_file_xgb = open("./models_howell/model_xgb_howell.txt", "r")
    accuracy_xgb = float(accuracy_file_xgb.read()) * 100

    lgb_model = pickle.load(open("./models_howell/model_lgb_howell.dat", "rb"))
    accuracy_file_lgb = open("./models_howell/model_lgb_howell.txt", "r")
    accuracy_lgb = float(accuracy_file_lgb.read()) * 100

    lda_model = pickle.load(open("./models_howell/model_lda_howell.dat", "rb"))
    accuracy_file_lda = open("./models_howell/model_lda_howell.txt", "r")
    accuracy_lda = float(accuracy_file_lda.read()) * 100

    return xgb_model, accuracy_xgb, lgb_model, accuracy_lgb, lda_model, accuracy_lda


(
    xgb_model_goldman,
    accuracy_xgb_model_goldman,
    lgb_model_goldman,
    accuracy_lgb_model_goldman,
    lda_model_goldman,
    accuracy_lda_model_goldman,
) = load_goldman_models()

# Goldman independent variables needed for the DataFrames

columns_goldman = ["BIB", "HML", "HHD", "RML", "FML", "FBL", "FHD", "TML", "FEB", "TPB", "HEB"]

# Loading the Howells models and their accuracy

(
    xgb_model_howell,
    accuracy_xgb_model_howell,
    lgb_model_howell,
    accuracy_lgb_model_howell,
    lda_model_howell,
    accuracy_lda_model_howell,
) = load_howell_models()

# Howell independent variables needed for the DataFrames

columns_howell = [
    "GOL",
    "NOL",
    "BNL",
    "BBH",
    "XCB",
    "XFB",
    "ZYB",
    "AUB",
    "WCB",
    "ASB",
    "BPL",
    "NPH",
    "NLH",
    "JUB",
    "NLB",
    "MAB",
    "MDH",
    "MDB",
    "OBH",
    "OBB",
    "DKB",
    "ZMB",
    "FMB",
    "EKB",
    "IML",
    "XML",
    "WMH",
    "STB",
    "FRC",
    "PAC",
    "OCC",
    "FOL",
]

title = """
<h1 style = "padding-top:0px; padding-bottom:0px; text-align: center; margin: 10px;" >
SexEst: A sex estimation web-application
</h1>
"""

st.markdown(title, unsafe_allow_html=True)

welcome_text = """
<p class="sexest-body">
<style>
p.sexest-body {text-align: justify; margin: 10px}
</style>

<hr style = "height:5px; border:none; color:#333; background-color:#333; margin: 10px;" />


Welcome to SexEst, a free, interactive, web application designed
to estimate sex using cranial or postcranial linear measurements.
Users can either enter manually the measurements for single skeletons
or upload data for multiple skeletons stored in a CSV file.
Sex estimation is based on three different machine learning
classification algorithms:
[Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) (LDA),
[Extreme Gradient Boosting](https://xgboost.readthedocs.io/en/stable/) (XGB) and
[Light Gradient Boosting](https://lightgbm.readthedocs.io/en/latest/) (LGB).
The training [datasets](http://volweb.utk.edu/~auerbach/DATA.htm)
used in these machine learning classifiers are
the William W. Howells craniometric dataset (Howells 1973, 1989, 1995)
for cranial measurements and the Goldman dataset (Auerbach and Ruff 2004, 2006)
for postcranial measurements. Both datasets include thousands of
individuals from various geographic locations dating throughout the
Holocene, hence they represent several broad geographic ancestral
backgrounds and account for inter-population variability in sexual
dimorphism. SexEst can generate a prediction even when a single variable
is given; hence, it is applicable even on highly fragmented remains or
remains where not all measurements can be accurately obtained due to
pathological or other alterations.


Instructions on how to use SexEst can be found by pressing the
**How to** button, while the contact details of the creators of the
application appear when pressing the **Contact** button.


SexEst was supported by the NI4OS-Europe project, funded by the
European Commission under the Horizon 2020 European Research
Infrastructures grant agreement no. 857645. EN's contribution was
co-funded by the European Regional Development Fund and the
Republic of Cyprus through the Research and Innovation Foundation
[Project: EXCELLENCE/1216/0023]. This project has also received
funding from the European Union's Horizon 2020 Research and
Innovation Program under the grant agreement no. 811068.


**Disclaimer:** This application is freely provided as an aid for
skeletal sex estimation. The authors hold no responsibility for
its ultimate use or misuse. As creators we try to ensure that the
software is theoretically grounded and statistically accurate, we
provide no warranty and make no specific claims as to its
performance or its appropriateness for use in any
particular situation.

**References**

Auerbach BM, Ruff CB. 2004. Human body mass estimation: A comparison of “morphometric” and “mechanical” methods. American Journal of Physical Anthropology 125: 331-342.
DOI: [10.1002/ajpa.20032](https://doi.org/10.1002/ajpa.20032)


Auerbach BM, Ruff CB. 2006. Limb bone bilateral asymmetry: variability and commonality among modern humans. Journal of
Human Evolution 50: 203-218.
DOI: [10.1016/j.jhevol.2005.09.004](https://doi.org/10.1016/j.jhevol.2005.09.004)


Howells WW. 1973. Cranial Variation in Man: A Study by Multivariate Analysis of Patterns of Difference among Recent Human Populations. Papers of the Peabody Museum of Archaeology and Ethnology, vol. 67: Harvard University, Cambridge, Mass.


Howells WW. 1989. Skull Shapes and the Map: Craniometric Analysis in the Dispersion of Modern Homo. Papers of the Peabody Museum of Archaeology and Ethnology, vol. 79: Harvard University, Cambridge, Mass.


Howells WW. 1995. Who's Who in Skulls. Ethnic Identification of Crania from Measurements. Papers of the Peabody Museum of Archaeology and Ethnology, vol. 82: Harvard University, Cambridge, Mass.




<hr style = "height:5px; border:none; color: #333; background-color:#333; margin: 10px;" />

[Paper](https://doi.org/10.1002/oa.3109) · [GitHub](https://github.com/cconsta1/SexEst) · [Notebooks](https://github.com/cconsta1/SexEst_Notebooks) · [Apache-2.0 License](https://github.com/cconsta1/SexEst/blob/master/LICENSE)

</p>
"""

welcome_citation = (
    "Constantinou C, Nikita E. 2022. SexEst: An open access web\n"
    "application for metric skeletal sex estimation.\n"
    "International Journal of Osteoarchaeology 32: 832-844.\n"
    "DOI: 10.1002/oa.3109"
)

padding = 0.98

st.markdown(
    f""" <style>
    div.block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """,
    unsafe_allow_html=True,
)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Navigation / view state
#
# st.session_state.view drives which of the four placeholders below get
# populated on any given rerun, so unrelated interactions (e.g. moving a
# sidebar slider) don't clobber whatever the user was last looking at.

if "view" not in st.session_state:
    st.session_state.view = "welcome"

if "result_payload" not in st.session_state:
    st.session_state.result_payload = None

if "warning_payload" not in st.session_state:
    st.session_state.warning_payload = None


def go_to(view_name, result_payload=None, warning_payload=None):
    st.session_state.view = view_name
    st.session_state.result_payload = result_payload
    st.session_state.warning_payload = warning_payload


placeholder_write_welcome = st.empty()
placeholder_write_input = st.empty()
placeholder_write_result = st.empty()
placeholder_write_warning = st.empty()

contact_section = """
<p class="sexest-body">
<style>
p.sexest-body {text-align: justify; margin: 10px; width: 744px}
</style>

<hr style = "height:5px; border:none; color:#333; background-color:#333; margin: 10px;" />

[Efthymia Nikita](https://cyi.academia.edu/EfthymiaNikita),
Assistant Professor in Bioarchaeology, Science and Technology in Archaeology
and Culture Research Center (STARC), The Cyprus Institute (<a href="mailto:e.nikita@cyi.ac.cy">e.nikita@cyi.ac.cy</a>)

[Chrysovalantis Constantinou](https://scholar.google.com/citations?user=b3hMHhAAAAAJ&hl=en), Computational Scientist, Computation-based Science and Technology Research Center (CaSToRC), The Cyprus Institute (<a href="mailto:cconsta1@alumni.nd.edu">cconsta1@alumni.nd.edu</a>)

<hr style = "height:5px; border:none; color:#333; background-color:#333; margin: 10px;" />

</p>
"""

how_to_section = """
<p class="sexest-body">
<style>
p.sexest-body {text-align: justify; margin: 10px}
</style>

<hr style = "height:5px; border:none; color:#333; background-color:#333; margin: 10px;" />

SexEst has three modes of operation: a single skeleton mode, a
multiple skeleton mode, and a missing data mode. For each mode
of operation, the application accepts either postcranial or cranial measurements.


Selecting one of the two single skeleton modes, **Osteometric Prediction (single skeleton)**
or **Craniometric Prediction (single skeleton)**,
reveals input boxes where the user can manually enter the postcranial or cranial
measurements. All measurements must be in millimeters (mm). The plus and minus
buttons allow the user to increase or decrease an input measurement by 0.5 mm.
Once the measurements are input, the machine learning model can be selected
among three available options: Extreme Gradient Boosting (XGB), Light Gradient
Boosting (LGB), and Linear Discriminant Analysis (LDA). Upon pressing the
Calculate button, the sex of the individual is estimated along with the
probability of being male or female; the accuracy of the model on which the
prediction was based is also given. Note that these modes require all
measurements to be present in order for the models to run; that is, they
cannot handle missing data. In cases of missing data, SexEst provides
alternative modes (see details below).


Selecting one of the multiple skeleton modes from the sidebar, **Osteometric Prediction (multiple skeletons)**
or **Craniometric Prediction (multiple skeletons)**,
allows users to upload a CSV file containing measurements from multiple
skeletons so that sex is predicted for all of them simultaneously.
The file format must follow the example files given at the bottom of this page.
The file header must contain the names of the 11 postcranial variables or the
32 cranial variables specified in the example files. All measurements must be in
millimeters (mm). The CSV file can be uploaded using the drag and
drop feature or by
browsing the computer's file system. Subsequently, the user
must select a model (XGB, LGB, LDA), and press the Calculate button.
The application will output the predicted result as a table divided into a
Male and a Female column containing the probability that each skeleton in the
file belongs to a male or a female individual. Note that any rows containing
missing data will be dropped as the models are optimized to use all 11 variables
for osteometric and all 32 variables for craniometric datasets, respectively.
In cases of missing data, SexEst provides alternative modes (see details below).


The missing data modes, **Osteometric Prediction (missing data)** or **Craniometric
Prediction (missing data)**, offer the possibility to make predictions when
one or more variables are missing. These tabs are similar to the single
skeleton modes; however, the user can now enter even a single variable and
get a sex prediction. For postcranial data, the user can enter any combination
of one, two, etc. out of the 11 measurements, while for the craniometric data we
have selected 10 out of the original 32 variables, which are most commonly
employed in other studies of population-specific metric sex estimation, namely
**GOL, BNL, BBH, XCB, ZYB, BPL, NLH, NLB, MDH,** and **FOL**. All entered measurements
must be in millimeters (mm). As these modes are more computationally intensive,
they have been trained using only Linear Discriminant Analysis.

If you encounter any problems, please contact the creators of the
application using the email addresses provided in the **Contact** tab.


<hr style = "height:5px; border:none; color:#333; background-color:#333; margin: 10px;" />


</p>
"""

osteometric_sample_dataframe = pd.read_csv(
    "sample_dataset_osteometric.csv", header=0, encoding="unicode_escape"
)
osteometric_sample_csv = osteometric_sample_dataframe.to_csv(index=False)

craniometric_sample_dataframe = pd.read_csv(
    "sample_dataset_craniometric.csv", header=0, encoding="unicode_escape"
)
craniometric_sample_csv = craniometric_sample_dataframe.to_csv(index=False)


def render_sample_download_buttons(key_prefix):
    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            label="Download sample osteometric csv file",
            data=osteometric_sample_csv,
            file_name="sample_dataset_osteometric.csv",
            mime="text/csv",
            key=f"{key_prefix}_osteometric_sample_download",
        )

    with download_col2:
        st.download_button(
            label="Download sample craniometric csv file",
            data=craniometric_sample_csv,
            file_name="sample_dataset_craniometric.csv",
            mime="text/csv",
            key=f"{key_prefix}_craniometric_sample_download",
        )


def render_result_payload(payload):
    kind = payload.get("kind")

    if kind == "single":
        placeholder_write_input.dataframe(payload["input_df"])
        placeholder_write_result.success(payload["result_text"])

    elif kind == "file":
        placeholder_write_input.dataframe(payload["input_df"])
        placeholder_write_result.dataframe(payload["result_df"])
        placeholder_write_warning.success(payload["info_text"])


if st.sidebar.button("Home", key="home_button"):
    go_to("welcome")

if st.sidebar.button("How to", key="how_to_button"):
    go_to("how_to")

if st.sidebar.button("Contact", key="contact_button"):
    go_to("contact")


with st.sidebar.expander("Osteometric Prediction (single skeleton)"):
    BIB = st.number_input("BIB", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    HML = st.number_input("HML", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    HHD = st.number_input("HHD", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    RML = st.number_input("RML", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    FML = st.number_input("FML", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    FBL = st.number_input("FBL", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    FHD = st.number_input("FHD", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    TML = st.number_input("TML", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    FEB = st.number_input("FEB", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    TPB = st.number_input("TPB", min_value=0.0, max_value=599.0, value=0.0, step=0.5)
    HEB = st.number_input("HEB", min_value=0.0, max_value=599.0, value=0.0, step=0.5)

    input_vector_goldman = [[BIB, HML, HHD, RML, FML, FBL, FHD, TML, FEB, TPB, HEB]]

    input_vector_goldman = np.asarray(input_vector_goldman, dtype="float64")

    is_all_zero_osteometric = not input_vector_goldman.all()

    models_goldman_list = ["<Not selected>", "XGB", "LGB", "LDA"]
    default_goldman = models_goldman_list.index("<Not selected>")

    model_selection_goldman = st.selectbox(
        "Select a model",
        options=models_goldman_list,
        index=default_goldman,
        key="model_selection_goldman",
    )

    if st.button("Calculate", key="osteometric_single_entry_button"):
        goldman_df = pd.DataFrame(input_vector_goldman, columns=columns_goldman)

        goldman_df = goldman_df.round(2)
        goldman_df = goldman_df.astype(str)

        if model_selection_goldman == "<Not selected>" and is_all_zero_osteometric:
            go_to(
                "result",
                warning_payload="""
                Some variables are missing and no model is selected. Please enter
                your variables and select a model.

                Please note that these models have been optimized to work with
                all 11 variables submitted. You can try
                the **Osteometric Prediction (missing data)** mode
                if you have missing data.
                """,
            )

        if model_selection_goldman != "<Not selected>" and is_all_zero_osteometric:
            go_to(
                "result",
                warning_payload="""
                Some variables are missing.

                Please note that these models have been optimized to work with
                all 11 variables submitted. You can try
                the **Osteometric Prediction (missing data)** mode
                if you have missing data.
                """,
            )

        if model_selection_goldman == "<Not selected>" and not is_all_zero_osteometric:
            go_to(
                "result",
                warning_payload="""
                Please select a model.
                """,
            )

        if model_selection_goldman != "<Not selected>" and not is_all_zero_osteometric:

            if model_selection_goldman == "XGB":
                predict_proba_goldman = xgb_model_goldman.predict_proba(
                    input_vector_goldman
                )

                accuracy_goldman = accuracy_xgb_model_goldman

            if model_selection_goldman == "LGB":
                predict_proba_goldman = lgb_model_goldman.predict_proba(
                    input_vector_goldman
                )

                accuracy_goldman = accuracy_lgb_model_goldman

            if model_selection_goldman == "LDA":
                predict_proba_goldman = lda_model_goldman.predict_proba(
                    input_vector_goldman
                )

                accuracy_goldman = accuracy_lda_model_goldman

            predict_proba_goldman = 100 * predict_proba_goldman

            predict_proba_goldman = predict_proba_goldman[0].tolist()

            result_text = """
            ###### The probability of the individual being male is {male:.2f}% and the probability of  being female is {female:.2f}%

            __The model was trained using 1528 cases. The dataset was
                split into a training set and a test set with
                proportions 70% and 30%, respectively. The model was then
                trained using the training
                set and [GridSearchCV](https://bit.ly/3F8s50E),
                which optimized the model's
                hyperparameters and cross-validated it. The trained
                model was then tested using the test set,
                achieving an accuracy of {accuracy:.2f}%.__
            """.format(
                male=predict_proba_goldman[0],
                female=predict_proba_goldman[1],
                accuracy=accuracy_goldman,
            )

            go_to(
                "result",
                result_payload={
                    "kind": "single",
                    "input_df": goldman_df,
                    "result_text": result_text,
                },
            )


with st.sidebar.expander("Craniometric Prediction (single skeleton)"):
    GOL = st.number_input("GOL", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    NOL = st.number_input("NOL", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    BNL = st.number_input("BNL", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    BBH = st.number_input("BBH", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    XCB = st.number_input("XCB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    XFB = st.number_input("XFB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    ZYB = st.number_input("ZYB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    AUB = st.number_input("AUB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    WCB = st.number_input("WCB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    ASB = st.number_input("ASB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    BPL = st.number_input("BPL", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    NPH = st.number_input("NPH", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    NLH = st.number_input("NLH", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    JUB = st.number_input("JUB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    NLB = st.number_input("NLB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    MAB = st.number_input("MAB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    MDH = st.number_input("MDH", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    MDB = st.number_input("MDB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    OBH = st.number_input("OBH", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    OBB = st.number_input("OBB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    DKB = st.number_input("DKB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    ZMB = st.number_input("ZMB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    FMB = st.number_input("FMB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    EKB = st.number_input("EKB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    IML = st.number_input("IML", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    XML = st.number_input("XML", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    WMH = st.number_input("WMH", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    STB = st.number_input("STB", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    FRC = st.number_input("FRC", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    PAC = st.number_input("PAC", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    OCC = st.number_input("OCC", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
    FOL = st.number_input("FOL", min_value=0.0, max_value=500.0, value=0.0, step=0.5)

    input_vector_howell = [
        [
            GOL,
            NOL,
            BNL,
            BBH,
            XCB,
            XFB,
            ZYB,
            AUB,
            WCB,
            ASB,
            BPL,
            NPH,
            NLH,
            JUB,
            NLB,
            MAB,
            MDH,
            MDB,
            OBH,
            OBB,
            DKB,
            ZMB,
            FMB,
            EKB,
            IML,
            XML,
            WMH,
            STB,
            FRC,
            PAC,
            OCC,
            FOL,
        ]
    ]

    input_vector_howell = np.asarray(input_vector_howell, dtype="float64")

    is_all_zero_craniometric = not input_vector_howell.all()

    models_howell_list = ["<Not selected>", "XGB", "LGB", "LDA"]
    default_howell = models_howell_list.index("<Not selected>")

    model_selection_howell = st.selectbox(
        "Select a model",
        options=models_howell_list,
        index=default_howell,
        key="model_selection_howell",
    )

    if st.button("Calculate", key="craniometric_single_entry_button"):
        howell_df = pd.DataFrame(input_vector_howell, columns=columns_howell)

        howell_df = howell_df.round(2)
        howell_df = howell_df.astype(str)

        if model_selection_howell == "<Not selected>" and is_all_zero_craniometric:
            go_to(
                "result",
                warning_payload="""
                Some variables are missing and no model is selected. Please enter
                your variables and select a model.

                Please note that these models have been optimized to work with
                all 32 variables submitted. You can try
                the **Craniometric Prediction (missing data)** mode
                if you have missing data.
                """,
            )

        if model_selection_howell != "<Not selected>" and is_all_zero_craniometric:
            go_to(
                "result",
                warning_payload="""
                Some variables are missing.

                Please note that these models have been optimized to work with
                all 32 variables submitted. You can try
                the **Craniometric Prediction (missing data)** mode
                if you have missing data.
                """,
            )

        if model_selection_howell == "<Not selected>" and not is_all_zero_craniometric:
            go_to(
                "result",
                warning_payload="""
                Please select a model.
                """,
            )

        if model_selection_howell != "<Not selected>" and not is_all_zero_craniometric:

            if model_selection_howell == "XGB":
                predict_proba_howell = xgb_model_howell.predict_proba(
                    input_vector_howell
                )

                accuracy_howell = accuracy_xgb_model_howell

            if model_selection_howell == "LGB":
                predict_proba_howell = lgb_model_howell.predict_proba(
                    input_vector_howell
                )

                accuracy_howell = accuracy_lgb_model_howell

            if model_selection_howell == "LDA":
                predict_proba_howell = lda_model_howell.predict_proba(
                    input_vector_howell
                )

                accuracy_howell = accuracy_lda_model_howell

            predict_proba_howell = 100 * predict_proba_howell

            predict_proba_howell = predict_proba_howell[0].tolist()

            result_text = """
            ###### The probability of the individual being male is {male:.2f}% and the probability of being female is {female:.2f}%

            _The model was trained using 3048 cases. The dataset was
                split into a training set and a test set with
                proportions 70% and 30%, respectively. The model was then
                trained using the training
                set and [GridSearchCV](https://bit.ly/3F8s50E),
                which optimized the model's
                hyperparameters and cross-validated it. The trained
                model was then tested using the test set,
                achieving an accuracy of {accuracy:.2f}%._
            """.format(
                male=predict_proba_howell[0],
                female=predict_proba_howell[1],
                accuracy=accuracy_howell,
            )

            go_to(
                "result",
                result_payload={
                    "kind": "single",
                    "input_df": howell_df,
                    "result_text": result_text,
                },
            )

with st.sidebar.expander("Osteometric Prediction (multiple skeletons)"):

    df_goldman_file = pd.DataFrame()
    goldman_file_upload_error = False

    uploaded_file_goldman = st.file_uploader(
        "Choose a CSV file", type=["csv"], key="osteometric_uploader"
    )

    if uploaded_file_goldman is not None:
        try:
            df_goldman_file = pd.read_csv(
                uploaded_file_goldman, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            )
            if list(df_goldman_file.columns[:11]) != columns_goldman and set(
                columns_goldman
            ) - set(df_goldman_file.columns) != set():
                goldman_file_upload_error = True
        except ValueError:
            goldman_file_upload_error = True

    models_goldman_file_list = ["<Not selected>", "XGB", "LGB", "LDA"]
    default_goldman_file = models_goldman_file_list.index("<Not selected>")

    model_selection_goldman_file = st.selectbox(
        "Select a model",
        options=models_goldman_file_list,
        index=default_goldman_file,
        key="model_selection_goldman_file",
    )

    if st.button("Calculate", key="osteometric_file_entry_button"):

        if goldman_file_upload_error:
            go_to(
                "upload_error",
                warning_payload="""
            Please check that your file's format complies with the example file format provided
            in the **How to** section. The header must contain the 11 postcranial
            variable names shown below, and you can grab a correctly formatted
            sample file using the download button underneath.
            """,
            )
            st.session_state.upload_error_kind = "osteometric"

        elif df_goldman_file.empty == True and (
            model_selection_goldman_file != "<Not selected>"
        ):
            go_to("result", warning_payload="Please upload a file")

        elif df_goldman_file.empty == True and (
            model_selection_goldman_file == "<Not selected>"
        ):
            go_to("result", warning_payload="Please upload a file and choose a model")

        elif (df_goldman_file.empty != True) and (
            model_selection_goldman_file == "<Not selected>"
        ):
            go_to("result", warning_payload="Please select a model")

        elif (df_goldman_file.empty != True) and (
            model_selection_goldman_file != "<Not selected>"
        ):

            df_goldman_file = df_goldman_file[(df_goldman_file != 0).all(1)]
            df_goldman_file = df_goldman_file.dropna()

            df_goldman_file.columns = columns_goldman
            df_goldman_file = df_goldman_file.round(2)
            df_goldman_file = df_goldman_file.astype(str)
            predict_proba_goldman_file = []
            accuracy_goldman_file = 0

            if model_selection_goldman_file == "XGB":
                predict_proba_goldman_file = xgb_model_goldman.predict_proba(
                    df_goldman_file.values
                )

                accuracy_goldman_file = accuracy_xgb_model_goldman

            if model_selection_goldman_file == "LGB":
                predict_proba_goldman_file = lgb_model_goldman.predict_proba(
                    df_goldman_file.values
                )

                accuracy_goldman_file = accuracy_lgb_model_goldman

            if model_selection_goldman_file == "LDA":
                predict_proba_goldman_file = lda_model_goldman.predict_proba(
                    df_goldman_file.values
                )

                accuracy_goldman_file = accuracy_lda_model_goldman

            predict_proba_goldman_file = pd.DataFrame(predict_proba_goldman_file * 100)

            predict_proba_goldman_file = predict_proba_goldman_file.round(2)

            predict_proba_goldman_file = predict_proba_goldman_file.astype(str)

            predict_proba_goldman_file.columns = ["Male", "Female"]

            info_text = """
                Please note that any rows containing missing data
                will be dropped as these models have been optimized to
                work with all 11 variables submitted. You can try the
                **Osteometric Prediction (missing data)** mode for
                any cases/rows containing missing data.


                _The model was trained using 1528 cases. The dataset was
                split into a training set and a test set with
                proportions 70% and 30%, respectively. The model was then
                trained using the training
                set and [GridSearchCV](https://bit.ly/3F8s50E),
                which optimized the model's
                hyperparameters and cross-validated it. The trained
                model was then tested using the test set,
                achieving an accuracy of {accuracy:.2f}%._
                """.format(
                accuracy=accuracy_goldman_file,
            )

            go_to(
                "result",
                result_payload={
                    "kind": "file",
                    "input_df": df_goldman_file.reset_index(drop=True),
                    "result_df": predict_proba_goldman_file,
                    "info_text": info_text,
                },
            )


with st.sidebar.expander("Craniometric Prediction (multiple skeletons)"):

    df_howell_file = pd.DataFrame()
    howell_file_upload_error = False

    uploaded_file_howell = st.file_uploader(
        "Choose a CSV file", type=["csv"], key="craniometric_uploader"
    )

    if uploaded_file_howell is not None:
        try:
            df_howell_file = pd.read_csv(
                uploaded_file_howell,
                usecols=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                ],
            )
        except ValueError:
            howell_file_upload_error = True

    models_howell_file_list = ["<Not selected>", "XGB", "LGB", "LDA"]
    default_howell_file = models_howell_file_list.index("<Not selected>")

    model_selection_howell_file = st.selectbox(
        "Select a model",
        options=models_howell_file_list,
        index=default_howell_file,
        key="model_selection_howell_file",
    )

    if st.button("Calculate", key="craniometric_file_entry_button"):

        if howell_file_upload_error:
            go_to(
                "upload_error",
                warning_payload="""
            Please check that your file's format complies with the example file format provided
            in the **How to** section. The header must contain the 32 cranial
            variable names shown in the example files, and you can grab a correctly
            formatted sample file using the download button underneath.
            """,
            )
            st.session_state.upload_error_kind = "craniometric"

        elif df_howell_file.empty == True and (
            model_selection_howell_file != "<Not selected>"
        ):
            go_to("result", warning_payload="Please upload a file")

        elif df_howell_file.empty == True and (
            model_selection_howell_file == "<Not selected>"
        ):
            go_to("result", warning_payload="Please upload a file and choose a model")

        elif (df_howell_file.empty != True) and (
            model_selection_howell_file == "<Not selected>"
        ):
            go_to("result", warning_payload="Please select a model")

        elif (df_howell_file.empty != True) and (
            model_selection_howell_file != "<Not selected>"
        ):

            df_howell_file = df_howell_file[(df_howell_file != 0).all(1)]
            df_howell_file = df_howell_file.dropna()

            df_howell_file.columns = columns_howell
            df_howell_file = df_howell_file.round(2)
            df_howell_file = df_howell_file.astype(str)
            predict_proba_howell_file = []

            accuracy_howell_file = 0

            if model_selection_howell_file == "XGB":
                predict_proba_howell_file = xgb_model_howell.predict_proba(
                    df_howell_file.values
                )

                accuracy_howell_file = accuracy_xgb_model_howell

            if model_selection_howell_file == "LGB":
                predict_proba_howell_file = lgb_model_howell.predict_proba(
                    df_howell_file.values
                )

                accuracy_howell_file = accuracy_lgb_model_howell

            if model_selection_howell_file == "LDA":
                predict_proba_howell_file = lda_model_howell.predict_proba(
                    df_howell_file.values
                )

                accuracy_howell_file = accuracy_lda_model_howell

            predict_proba_howell_file = pd.DataFrame(predict_proba_howell_file * 100)

            predict_proba_howell_file = predict_proba_howell_file.round(2)

            predict_proba_howell_file = predict_proba_howell_file.astype(str)

            predict_proba_howell_file.columns = ["Male", "Female"]

            info_text = """
                Please note that any rows containing missing data will be
                dropped as these models have been optimized to work with
                all 32 variables submitted. You can try the
                **Craniometric Prediction (missing data)** mode for
                any cases/rows containing missing data.

                 _The model was trained using 3048 cases. The dataset was
                split into a training set and a test set with
                proportions 70% and 30%, respectively. The model was then
                trained using the training
                set and [GridSearchCV](https://bit.ly/3F8s50E),
                which optimized the model's
                hyperparameters and cross-validated it. The trained
                model was then tested using the test set,
                achieving an accuracy of {accuracy:.2f}%._
                """.format(
                accuracy=accuracy_howell_file,
            )

            go_to(
                "result",
                result_payload={
                    "kind": "file",
                    "input_df": df_howell_file.reset_index(drop=True),
                    "result_df": predict_proba_howell_file,
                    "info_text": info_text,
                },
            )


with st.sidebar.expander("Osteometric Prediction (missing data)"):
    BIB = st.number_input("BIB", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    HML = st.number_input("HML", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    HHD = st.number_input("HHD", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    RML = st.number_input("RML", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    FML = st.number_input("FML", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    FBL = st.number_input("FBL", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    FHD = st.number_input("FHD", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    TML = st.number_input("TML", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    FEB = st.number_input("FEB", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    TPB = st.number_input("TPB", min_value=0.0, max_value=600.0, value=0.0, step=0.5)
    HEB = st.number_input("HEB", min_value=0.0, max_value=600.0, value=0.0, step=0.5)

    input_vector_goldman_missing = [[BIB, HML, HHD, RML, FML, FBL, FHD, TML, FEB, TPB, HEB]]

    input_vector_goldman_missing = np.asarray(
        input_vector_goldman_missing, dtype="float64"
    )

    model_files_list_goldman_missing = os.listdir("./models_goldman_missing_data")

    index_of_selected_features_goldman_missing = np.nonzero(
        input_vector_goldman_missing
    )[1].tolist()

    user_vector_goldman_missing = [
        columns_goldman[index] for index in index_of_selected_features_goldman_missing
    ]

    user_vector_goldman_missing.append(".dat")

    target_file_goldman_missing = [
        x
        for x in model_files_list_goldman_missing
        if set(user_vector_goldman_missing) == set(x.split("_")[3::])
    ]

    input_dataframe_goldman_missing = pd.DataFrame(
        data=input_vector_goldman_missing, columns=columns_goldman
    )

    user_vector_goldman_missing = user_vector_goldman_missing[:-1]

    input_dataframe_goldman_missing = input_dataframe_goldman_missing[
        user_vector_goldman_missing
    ]

    if st.button("Calculate", key="osteometric_single_entry_button_missing_data"):

        predict_proba_goldman_missing = 0

        if not input_vector_goldman_missing.any():
            go_to(
                "result",
                warning_payload="""
                Please enter a variable.
                """,
            )

        else:

            path_goldman_missing = (
                "./models_goldman_missing_data/" + target_file_goldman_missing[0]
            )

            path_accuracy_goldman_missing = (
                "./models_goldman_missing_data/"
                + target_file_goldman_missing[0][:-4]
                + ".txt"
            )

            model_goldman_missing = pickle.load(open(path_goldman_missing, "rb"))

            accuracy_file_goldman_missing = open(path_accuracy_goldman_missing, "r")

            accuracy_goldman_missing = float(accuracy_file_goldman_missing.read()) * 100

            predict_proba_goldman_missing = model_goldman_missing.predict_proba(
                input_dataframe_goldman_missing.values
            )

            predict_proba_goldman_missing = 100 * predict_proba_goldman_missing

            predict_proba_goldman_missing = predict_proba_goldman_missing[0].tolist()

            input_dataframe_goldman_missing = input_dataframe_goldman_missing.round(2)
            input_dataframe_goldman_missing = input_dataframe_goldman_missing.astype(
                str
            )

            result_text = """
            ###### The probability of the individual being male is {male:.2f}% and the probability of being female is {female:.2f}%

            _The model was trained using 1528 ({vars}) cases. The dataset was
                split into a training set and a test set with
                proportions 70% and 30%, respectively. The model was then
                trained using the training
                set and [GridSearchCV](https://bit.ly/3F8s50E),
                which optimized the model's
                hyperparameters and cross-validated it. The trained
                model was then tested using the test set,
                achieving an accuracy of {accuracy:.2f}%._
            """.format(
                male=predict_proba_goldman_missing[0],
                female=predict_proba_goldman_missing[1],
                vars=", ".join(user_vector_goldman_missing),
                accuracy=accuracy_goldman_missing,
            )

            go_to(
                "result",
                result_payload={
                    "kind": "single",
                    "input_df": input_dataframe_goldman_missing,
                    "result_text": result_text,
                },
            )


with st.sidebar.expander("Craniometric Prediction (missing data)"):
    GOL = st.number_input("GOL", min_value=0.0, max_value=580.0, value=0.0, step=0.5)
    BNL = st.number_input("BNL", min_value=0.0, max_value=580.0, value=0.0, step=0.5)
    BBH = st.number_input("BBH", min_value=0.0, max_value=580.0, value=0.0, step=0.5)
    XCB = st.number_input("XCB", min_value=0.0, max_value=580.0, value=0.0, step=0.5)
    ZYB = st.number_input("ZYB", min_value=0.0, max_value=580.0, value=0.0, step=0.5)
    BPL = st.number_input("BPL", min_value=0.0, max_value=580.0, value=0.0, step=0.5)
    NLH = st.number_input("NLH", min_value=0.0, max_value=580.0, value=0.0, step=0.5)
    NLB = st.number_input("NLB", min_value=0.0, max_value=580.0, value=0.0, step=0.5)
    MDH = st.number_input("MDH", min_value=0.0, max_value=580.0, value=0.0, step=0.5)
    FOL = st.number_input("FOL", min_value=0.0, max_value=580.0, value=0.0, step=0.5)

    columns_howell_missing = [
        "GOL",
        "BNL",
        "BBH",
        "XCB",
        "ZYB",
        "BPL",
        "NLH",
        "NLB",
        "MDH",
        "FOL",
    ]

    input_vector_howell_missing = [[GOL, BNL, BBH, XCB, ZYB, BPL, NLH, NLB, MDH, FOL]]

    input_vector_howell_missing = np.asarray(
        input_vector_howell_missing, dtype="float64"
    )

    model_files_list_howell_missing = os.listdir("./models_howell_missing_data")

    index_of_selected_features_howell_missing = np.nonzero(input_vector_howell_missing)[
        1
    ].tolist()

    user_vector_howell_missing = [
        columns_howell_missing[index]
        for index in index_of_selected_features_howell_missing
    ]

    user_vector_howell_missing.append(".dat")

    target_file_howell_missing = [
        x
        for x in model_files_list_howell_missing
        if set(user_vector_howell_missing) == set(x.split("_")[3::])
    ]

    input_dataframe_howell_missing = pd.DataFrame(
        data=input_vector_howell_missing, columns=columns_howell_missing
    )

    user_vector_howell_missing = user_vector_howell_missing[:-1]

    input_dataframe_howell_missing = input_dataframe_howell_missing[
        user_vector_howell_missing
    ]

    if st.button("Calculate", key="craniometric_single_entry_button_missing_data"):

        predict_proba_howell_missing = 0

        if not input_vector_howell_missing.any():
            go_to(
                "result",
                warning_payload="""
                Please enter a variable.
                """,
            )

        else:
            path_howell_missing = (
                "./models_howell_missing_data/" + target_file_howell_missing[0]
            )

            path_accuracy_howell_missing = (
                "./models_howell_missing_data/"
                + target_file_howell_missing[0][:-4]
                + ".txt"
            )

            model_howell_missing = pickle.load(open(path_howell_missing, "rb"))

            accuracy_file_howell_missing = open(path_accuracy_howell_missing, "r")

            accuracy_howell_missing = float(accuracy_file_howell_missing.read()) * 100

            predict_proba_howell_missing = model_howell_missing.predict_proba(
                input_dataframe_howell_missing.values
            )

            predict_proba_howell_missing = 100 * predict_proba_howell_missing

            predict_proba_howell_missing = predict_proba_howell_missing[0].tolist()

            input_dataframe_howell_missing = input_dataframe_howell_missing.round(2)
            input_dataframe_howell_missing = input_dataframe_howell_missing.astype(str)

            result_text = """
            ###### The probability of the individual being male is {male:.2f}% and the probability of being female is {female:.2f}%

            _The model was trained using 3048 ({vars}) cases. The dataset was
                split into a training set and a test set with
                proportions 70% and 30%, respectively. The model was then
                trained using the training
                set and [GridSearchCV](https://bit.ly/3F8s50E),
                which optimized the model's
                hyperparameters and cross-validated it. The trained
                model was then tested using the test set,
                achieving an accuracy of {accuracy:.2f}%._
            """.format(
                male=predict_proba_howell_missing[0],
                female=predict_proba_howell_missing[1],
                vars=", ".join(user_vector_howell_missing),
                accuracy=accuracy_howell_missing,
            )

            go_to(
                "result",
                result_payload={
                    "kind": "single",
                    "input_df": input_dataframe_howell_missing,
                    "result_text": result_text,
                },
            )


# Single render pass, driven entirely by st.session_state.view, so that
# whichever screen the user is on survives unrelated reruns (e.g. moving
# a sidebar number_input) instead of being silently reset to the welcome
# screen or cleared.

current_view = st.session_state.view

placeholder_write_welcome.empty()
placeholder_write_input.empty()
placeholder_write_result.empty()
placeholder_write_warning.empty()

if current_view == "welcome":
    placeholder_write_welcome.markdown(welcome_text, unsafe_allow_html=True)
    with placeholder_write_input.container():
        st.markdown("**Cite this work**")
        st.code(welcome_citation, language=None)

elif current_view == "how_to":
    with placeholder_write_welcome.container():
        st.markdown(how_to_section, unsafe_allow_html=True)
        render_sample_download_buttons(key_prefix="how_to")

elif current_view == "contact":
    placeholder_write_welcome.markdown(contact_section, unsafe_allow_html=True)

elif current_view == "upload_error":
    with placeholder_write_warning.container():
        st.warning(st.session_state.warning_payload)
        upload_error_kind = st.session_state.get("upload_error_kind")
        if upload_error_kind == "osteometric":
            st.download_button(
                label="Download sample osteometric csv file",
                data=osteometric_sample_csv,
                file_name="sample_dataset_osteometric.csv",
                mime="text/csv",
                key="upload_error_osteometric_sample_download",
            )
        elif upload_error_kind == "craniometric":
            st.download_button(
                label="Download sample craniometric csv file",
                data=craniometric_sample_csv,
                file_name="sample_dataset_craniometric.csv",
                mime="text/csv",
                key="upload_error_craniometric_sample_download",
            )

elif current_view == "result":
    if st.session_state.warning_payload is not None:
        placeholder_write_warning.warning(st.session_state.warning_payload)
    if st.session_state.result_payload is not None:
        render_result_payload(st.session_state.result_payload)
