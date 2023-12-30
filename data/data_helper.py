from ast import Delete
from asyncio.windows_events import NULL
import streamlit as st
from sklearn.datasets import load_iris,load_diabetes
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt



data_dict={
 "Concentric Circles":pd.read_csv('data/datasets/concertriccir2.csv'),
"Outlier":pd.read_csv('data/datasets/outlier.csv'),
"Two Spirals":pd.read_csv('data/datasets/twoSpirals.csv'),
"U-Shaped":pd.read_csv('data/datasets/ushape.csv'),
"XOR":pd.read_csv('data/datasets/xor.csv'),
"Linearly Separable":pd.read_csv('data/datasets/linearsep.csv')
}





class Datasets:

    

    def __init__(self):
        pass



    def load_data(self,data):
        self.data=data
        df=data_dict[data]

        # X=df["X"]
        # y=df["Y"]

        with st.sidebar :
            
            st.write("Shape :" , df.shape)

        st.write(f"{data} Dataset")

        
        
        with st.expander("See The DataFrame Here"):
            st.dataframe(df.head(5))
        
            
        return df

        

    

    
