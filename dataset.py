import streamlit as st
import yfinance as yf

from datetime import date

st.title("Dataset Minyak Mentah Brent")
st.write("Dataset harga diambil dari situs [Yahoo Finance](https://finance.yahoo.com/quote/BZ%3DF/history?p=BZ%3DF).")

START = "2012-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
data = yf.download('BZ=F', START, TODAY)
data.reset_index(inplace=True)

st.subheader('Raw data')
st.write(data.tail())

st.subheader('Plot Data Kolom Close pada Harga Harian Brent Crude Oil Price')
st.line_chart(data.Close)