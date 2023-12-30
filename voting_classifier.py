from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt



class Voting:

    def __init__(self) -> None:

        pass



    

    def preprocess(self,X,y):
        self.X=X
        self.y=y
        chart_data = X

        # with st.sidebar:
        #     st.write("Scatter plot Features")
        #     chosen_col1=st.selectbox("choose Column 1",(chart_data.columns),key="col1")
        #     chosen_col2=st.selectbox("choose Column 1",(chart_data.columns),key="col2")

        #using st.altair_chart
        c = (
        alt.Chart(chart_data)
        .mark_circle()
        .encode( )
        .encode(x="X", y="Y", size="X", color="Y")
        )
        
        with st.expander("See The Scatterplot",expanded=True):
            st.altair_chart(c, use_container_width=True)


    def models(self,X,y):
        
    

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

        return X_train,X_test,y_train,y_test 



