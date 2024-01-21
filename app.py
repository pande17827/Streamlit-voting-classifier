from multiprocessing.sharedctypes import Value
import streamlit as st 
st.set_page_config(initial_sidebar_state="expanded",layout="wide",)

from data.data_helper import Datasets

from voting_classifier import Voting

from run_algo import Run
import pandas as pd

import pkg_resources

def display_installed_packages():
    # Get a list of installed packages and their versions
    installed_packages = [f"{dist.project_name}=={dist.version}" for dist in pkg_resources.working_set]

    # Display the result in a Streamlit text area
    st.text_area("Installed Packages:", "\n".join(installed_packages), height=400)

display_installed_packages()



st.title('Voting Classifier Using Different Algos')

obj_data=Datasets()

vote=Voting()

with st.sidebar:
    st.title("Voting Classifier and More")

    data = st.sidebar.selectbox(
    "Dataset",
    ("XOR","Outlier","U-Shaped", "Linearly Separable", "Two Spirals","Concentric Circles")
)


    


df=obj_data.load_data(data)


X = df.iloc[:, :-1]
y=df.iloc[:,-1]

with st.sidebar:
    visualize=st.checkbox("Visualize?",value=False,)

if visualize:
    vote.preprocess(X,y)


X_train,X_test,y_train,y_test=vote.models(X,y)





voting_type = st.sidebar.radio(
        "Voting Type",
        (
        'hard',
        'soft',
        )
        )


estimators = st.sidebar.multiselect(
'Estimators',
[
    'KNN',
    'Logistic Regression',
    'Gaussian Naive Bayes',
    'SVM',
    'Random Forest'
],
default=["Logistic Regression"]
)


algo=Run(X_train,X_test,y_train,y_test,voting_type,estimators)

if st.sidebar.button("Run Algorithm"):
    
    accuracy,cross_val=algo.train_classifier()
    model_data=algo.each_model_accuracy()

    col1,col2=st.columns([1,1])
    with col1:
        st.write("Voting classifier Accuracy")
        st.write([f"Voting classifier :{accuracy}"])
        st.write([f"Cross Val Score :{cross_val}"])

    with col2:
        st.write("Each model Accuracy")
        st.write(model_data)


