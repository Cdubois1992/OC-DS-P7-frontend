import streamlit as st
import pandas as pd
import shap
import pickle


st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="Home")


# --- Initialising SessionState ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# --- Layout of the Home page ---

st.title("Prêt à dépenser")
st.subheader("Loan decision support app")

st.write("""This app assists the loan officer for his decision to grant a loan to a client.
     To do so, a machine learning algorithm is used to predict the difficulties of a client to repay the loan.
     For transparancy, this app also provide informations to explain the algorithm and predictions""")

col1, col2 = st.columns(2)

# --- Logo ---
with col1:
    st.image("image/logo.png")

# --- Pages description ---
with col2:

    st.write(" ")
    st.write(" ")  # Empty string to center the following text
    st.write(" ")

    st.subheader("Content of the App :")
    st.markdown("""
     This app includes three pages :
     1) General informations about the database and the model
     2) Analysis of known clients
     3) Prediction of loan default for new client through an API
     """)


# --- Loading data ---

st.subheader("App Loading: ")


with st.spinner('initialization...'):  # Show loading status
    @st.cache  # caching to improve performance and save outputs
    def loading_data():
        # Loading dataframes
        url = 'https://drive.google.com/file/d/1RMpmFw4F8sJ2aheTW6j7AsxjXirQCPU-/view?usp=sharing'
        path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
        df_train = pd.read_csv(
            path,
            sep=';',
            index_col="SK_ID_CURR")

        url= 'https://drive.google.com/file/d/1dM1nHOZTGnPF5nCAJwOkVJT9X6GwDkO3/view?usp=sharing'
        path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
        df_new = pd.read_csv(
            path,
            sep=';',
            index_col="SK_ID_CURR")
        return df_train, df_new

    st.write("1) Loading data")
    df_train, df_new = loading_data()

    st.write("2) Loading model")
    model = "model.pkl"
    Credit_clf_final = pickle.load(open(model, 'rb'))

    st.write("3) Loading Explainer (Shap) ")
    explainer = shap.TreeExplainer(
        Credit_clf_final, df_train.drop(
            columns="TARGET").fillna(0))

    st.write("4) Saving session variables")
    st.session_state.df_train = df_train
    st.session_state.df_new = df_new
    st.session_state.Credit_clf_final = Credit_clf_final
    st.session_state.explainer = explainer

    st.success('Done!')
