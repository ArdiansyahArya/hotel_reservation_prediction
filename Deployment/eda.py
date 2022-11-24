import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="Hotel Bookings Data",
    layout='wide',
    initial_sidebar_state='expanded'
)
def run(): # Agar bisa dipanggil oleh code main.
    # Membuat Title
    st.title('Hotel Bookings Data')

    # Membuat Sub Header
    st.subheader('EDA untuk Analisa Dataset Hotel Bookings Demand')

    # Membuat Deskripsi
    st.write('Page ini dibuat oleh *Ardiansyah Arya Salvinia')

    # Menambahkan Garis Lurus
    st.markdown('---')

    # Magic Syntax untuk menuliskan text beberapa baris
    '''
    Pada page kali ini, penulis akan melakukan eksplorasi sederhana.
    Dataset yang digunakan adalah dataset Hotel Booking Demand.
    Dataset ini berasal dari web www.kaggle.com.
    '''

    # Show Dataframe
    data = pd.read_csv('hotel_bookings.csv')
    st.dataframe(data)

    # Mendapatkan kolom numerical dan kolom kategorikal

    num_columns = data.select_dtypes(include=np.number).columns.tolist()
    cat_columns = data.select_dtypes(include=['object']).columns.tolist()


    # Membuat BarPlot
    st.write('### Plot Lead Time berbanding Pembatalan')
    plot = data[num_columns].groupby('is_canceled').mean().reset_index()
    

    fig = plt.figure(figsize=(10,10))
    plt.bar(x=plot['is_canceled'], height=plot['lead_time'])
    X_axis = np.arange(len(plot['is_canceled']))
    plt.xticks([0,1])
    plt.xlabel('is_canceled')
    st.pyplot(fig)

    # Membuat BarPlot
    st.write('### Plot Jumlah Hari Masuk Waiting List')
    fig = plt.figure(figsize=(10,10))
    plt.bar(x=plot['is_canceled'], height=plot['days_in_waiting_list'])
    plt.xticks(X_axis, plot['is_canceled'])
    plt.legend()
    st.pyplot(fig)

    week=data[{'is_canceled','arrival_date_week_number'}].value_counts().reset_index()

    
    st.write('### Plot Minggu Kedatangan')
    y1=week[{'arrival_date_week_number',0}].loc[week['is_canceled']==1].sort_values('arrival_date_week_number')
    y2=week[{'arrival_date_week_number',0}].loc[week['is_canceled']==0].sort_values('arrival_date_week_number')
    fig = plt.figure(figsize=(10,10))
    plt.plot(y1['arrival_date_week_number'],y1[0], label='cancelled')
    plt.plot(y2['arrival_date_week_number'],y2[0], label='not cancelled')
    plt.legend()
    st.pyplot(fig)

    corr=data[num_columns].corr()
    corr.drop('is_canceled', axis=0, inplace=True)
    st.dataframe(corr['is_canceled'].sort_values(ascending=False))



if __name__ == '__main__': # Agar python bisa dibuka standalone tanpa membuka main.
    run()