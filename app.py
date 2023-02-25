from ssl import Options
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from datetime import date
from streamlit_option_menu import option_menu



with st.sidebar:
	menu = option_menu(
	menu_title=None,
	icons=['house','archive','bar-chart'],
	options=["Home","Raw Dataset","Prediksi"],
	default_index=0
)

if menu == "Home":
	st.title(f"Prediksi Harga Minyak Mentah")
if menu == "Raw Dataset":
	st.title(f"Data Minyak Mentah West Texas Intermediate (WTI)")
	START = "2012-01-01"
	TODAY = date.today().strftime("%Y-%m-%d")
data_minyak = ('CL=F', 'BZ=F')
selected_data_minyak = st.selectbox('Select dataset for prediction', data_minyak)

n_days = st.slider('Days of prediction:', 1, 7)


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_data_minyak)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

if menu == "Prediksi":
	st.title(f"Peramalan Harga Minyak Mentah")


