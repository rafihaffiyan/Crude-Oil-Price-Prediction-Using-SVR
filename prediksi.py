import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from datetime import date

st.title("Prediksi harga minyak Brent dengan _Support Vector Regression_")
st.subheader("Raw Dataset")
START = ("2017-01-01")
TODAY = date.today().strftime("%Y-%m-%d")

data = yf.download('BZ=F', START, TODAY)
data.reset_index(inplace=True)

st.write(data.tail())

closedf = data[['Date','Close']]

close_stock = closedf.copy()
del closedf['Date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

st.subheader("Data Normalisasi Harga Penutupan")
st.write(closedf)

training_size=int(len(closedf)*0.8)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]  
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


st.subheader("Pilih Kernel")
pilih = st.radio( "Pilih Kernel yang akan digunakan:",
    ('rbf', 'linear', 'poly', 'sigmoid'))


model = SVR(kernel = pilih, C= 1e1, gamma= 1)
model.fit(X_train, y_train)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

st.write("Nilai MAPE: ", mean_absolute_percentage_error(original_ytest,test_predict))

from itertools import cycle
import plotly.express as px
look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

st.subheader("Plot data")
names = cycle(['Harga Close','Data Testing'])

plotdf = pd.DataFrame({'Date': close_stock['Date'],
                       'original_close': close_stock['Close'],
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['Date'], y=[plotdf['original_close'],
                                        plotdf['test_predicted_close']],
              labels={'value':'Harga Minyak Bumi','date': 'Date'})
fig.update_layout(title_text='Plot Data Aktual dan Prediksi',
                  plot_bgcolor='white', font_size=15, font_color='white',legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
st.write(fig)


dataprediksi = pd.DataFrame({'Date': plotdf['Date'],
                       'original_close': plotdf['original_close'],
                      'predicted_close': plotdf['test_predicted_close']})
akhir = dataprediksi.iloc[:-1 , :]
st.subheader("Perbandingan Data Aktual dan Data Prediksi")
st.write(akhir.tail(30))


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()



lst_output=[]
n_steps=time_step
i=0
pred_days = 7
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input=np.array(temp_input[1:])
        
        x_input=x_input.reshape(1,-1)
        
        yhat = model.predict(x_input)
        
        temp_input.extend(yhat.tolist())
        temp_input=temp_input[1:]
       
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        yhat = model.predict(x_input)
        
        temp_input.extend(yhat.tolist())
        lst_output.extend(yhat.tolist())
        
        i=i+1

last_days=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+pred_days+1)

temp_mat = np.empty((len(last_days)+pred_days+1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

nextpred = pd.DataFrame({
    'Prediksi':next_predicted_days_value
})

st.subheader("Hasil Prediksi 7 Hari Berikutnya")
st.write(nextpred.tail(7))

#st.subheader("Grafik Hasil Prediksi 7 Hari Berikutnya")
#st.line_chart(nextpred)
#new_pred_plot = pd.DataFrame({
#    'next_predicted_days_value':next_predicted_days_value
#})

#names = cycle(['Harga Prediksi di Hari Berikutnya'])

#fig = px.line(new_pred_plot,x=new_pred_plot.index, y=new_pred_plot['next_predicted_days_value'],
#              labels={'value': 'Harga','index': 'Rentang Waktu'})
#fig.update_layout(plot_bgcolor='white', font_size=15, font_color='white',legend_title_text='Close Price')
#fig.for_each_trace(lambda t:  t.update(name = next(names)))
#fig.update_xaxes(showgrid=False)
#fig.update_yaxes(showgrid=False)
#st.write(fig)



