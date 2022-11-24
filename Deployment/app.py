# Digunakan sebagai pusat code yang akan digunakan user.
import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilh Halaman : ', ('EDA', 'Prediksi Pembatalan Reservasi'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()