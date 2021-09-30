import warnings
warnings.filterwarnings('ignore')
import numpy as np
import streamlit as st
import sys
import csv
import pydeck as pdk
from sklearn.model_selection import train_test_split
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score
class EDA:
    def upload(self):
        uploaded_file=st.file_uploader("Choose a file")
        df=pd.read_csv(uploaded_file)
        df.dropna(inplace=True, axis=0)
        #st.dataframe(df.style.highlight_max(axis=0))
        data=df[['minimum_nights', 'number_of_reviews', 'availability_365', 'price', 'neighbourhood']]
        st.dataframe(data)
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        st.header('X Columns or Feature Columns ')
        st.write('Rows in x', x.shape)
        st.dataframe(x)
        st.header('Y columns or Target')
        st.write('Rows in y', y.shape)
        st.dataframe(y)
        st.header('NaN Value Target')
        st.write(data.isnull().sum())
        st.header('Data distribution plot')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.pairplot(data)
        st.pyplot()
        st.header('Correalation')
        st.write(data.corr())
        st.header('HEAT MAP')
        sns.heatmap(data.corr())
        st.pyplot()
        map_df=df[['latitude','longitude']]
        st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/sanjeevpandey/ckrf7it0z3hoz17o9pn17zxh6',initial_view_state=pdk.ViewState(latitude=40.730610, longitude=-73.935242, zoom=1,pitch=50)))
        st.map(map_df)  # plotting in map
        st.header('AIR BNB LISTING')
        st.write(df['neighbourhood_group'].value_counts())
        y = df['neighbourhood_group'].value_counts()
        st.bar_chart(y)
        return data,df
    @staticmethod
    def split(df):
        x=df.iloc[:, :-1]
        y=df.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
        return x_train, x_test, y_train, y_test
    @staticmethod
    def selection_model(x_train,y_train,minimum_nights,number_of_reviews,availability_365,price):
        model=DecisionTreeClassifier()  # create object of model
        model.fit(x_train, y_train)  # fit the model
        p=model.predict([[minimum_nights,number_of_reviews,availability_365,price]])
        st.write('Place in NEW-YORK:',p)
    @staticmethod
    def search(data,place):
        df=data
        df=df[['latitude', 'longitude','neighbourhood']]
        d = df.loc[df['neighbourhood'] == place]
        cor = d.head(1)
        st.write(cor)
        cord=cor.iloc[:, :-1]
        st.map(cord)



a=EDA()
if st.checkbox("EDA"):
    data,df=a.upload()
    x_train,x_test,y_train,y_test=EDA.split(data)
if st.checkbox("PREDICTION"):
    minimum_nights=st.number_input('Enter number of night',step=1)  # take input value from user
    number_of_reviews=st.number_input('Review',step=1)
    availability_365=st.number_input('avilability in year',step=1)
    price=st.number_input('PRICE', step=0.10)
    EDA.selection_model(x_train,y_train,minimum_nights,number_of_reviews,availability_365,price)
if st.checkbox("SEARCH AREA"):
    place=st.text_input('Enter area')
    EDA.search(df,place)
