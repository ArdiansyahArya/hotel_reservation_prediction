import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import joblib
import json
import datetime
import pickle
from CustomTransformer import OutletTypeEncoder

# Load All Files
with open('model_decision_tree.pkl', 'rb') as file_1:
  dc = joblib.load(file_1)

with open('model_scaler_skewed.pkl', 'rb') as file_2:
  scaler_skewed = joblib.load(file_2)

with open('model_scaler_normal.pkl', 'rb') as file_3:
  scaler_normal = joblib.load(file_3)

with open('model_encoder.pkl', 'rb') as file_4:
  encoder = joblib.load(file_4)

with open('list_num_normal.txt', 'r') as file_5:
  normal_dist = json.load(file_5)

with open('list_num_skewed.txt', 'r') as file_6:
  skewed_dist = json.load(file_6)

with open('list_cat_cols.txt', 'r') as file_7:
  cat_columns = json.load(file_7)

with open('list_drop_cols.txt', 'r') as file_8:
  kolom_drop = json.load(file_8)

num_n_pipeline = pickle.load(open('num_n_pipeline.pkl', 'rb'))

num_s_pipeline = pickle.load(open('num_s_pipeline.pkl', 'rb'))

preprocess_pipeline = pickle.load(open('preprocess_pipeline.pkl', 'rb'))

model_pipeline = pickle.load(open('model_pipeline.pkl', 'rb'))



def run(): # Agar bisa dipanggil oleh main.
    # Membuat Form
    with st.form(key='form_data_konsumen'):
        
        country = st.text_input('Negara Asal', max_chars=3, help= 'Di isi menggunakan standard ALpha-3 Code ISO-3166-1 (ref = `https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes`) ')
        customer_type = st.selectbox('Tipe Reservasi',('Contract','Group','Transient','Transient-party'), help = 'Contract = Bila tipe reservasi nya ada sebuah kontrak khusus, Group = Reservasi sekelompok orang, Transient = Booking yang tidak masuk ke kontrak atau group, Transient-party = Sekelompok reservasi tipe transient. ' )
        deposit_type = st.selectbox('Jenis Pembayaran Deposit', ('No Deposit','Non Refund', 'Refundable'), help = 'No Deposit = Konsumen belum membayar, Non Refund = Deposit telah dibayar sesuai dengan harga inap, Refundable = Deposit telah dibayar namun masih dibawah harga inap.')
        hotel = st.selectbox('Jenis Hotel', ('Resort Hotel','City Hotel')) # Value = nilai default
        is_repeated_guest = st.selectbox('Pernah Menginap di Hotel', (1 , 0) , help='1 = Pernah Menginap, 0 = Belum Pernah Menginap')
        meal = st.selectbox('Paket meal yang dipesan', ('SC','BB','HB','FB'), help = 'SC = Hanya pesan kamar, BB = Breakfast, HB = Breakfast + Lunch/Dinner, FB = Full Meal')
        market_segment = st.selectbox('Segmentasi Pasar Hotel', ('Groups','Corporate','Offline TA/TO', 'Online TA','Direct','Complementary','Aviation','Undefined'), help = 'TA = Travel Agents, TO = Tour Operators')
        distribution_channel = st.selectbox('Channel distribusi', ('TA/TO','Corporate','Dircet','GDS','Undefined'), help = 'TA = Travel Agents, TO = Tour Operators')
        reservation_status = st.selectbox('Status Reservasi', ('Canceled','Check-Out','No-Show'), help = 'Canceled = Reservasi dibatalkan, Check-Out = Customer sudah check out, No-Show = Tidak ada kabar')
        reservation_status_date = st.date_input('Tanggal Input Status Reservasi', value=datetime.date(2022,11,25))

        st.markdown('---')

        adults = st.number_input('Jumlah Orang Dewasa', min_value=1, max_value=99, value=1)
        children = st.number_input('Jumlah Anak-anak', min_value=0, max_value=99, value=0)
        babies = st.number_input('Jumlah Bayi', min_value=0, max_value=99, value=0)


        st.markdown('---')

        agent = st.number_input('Agent ID', min_value = 0, max_value = 999, value = 0, help = '0 = Bukan dari Agent')
        company = st.number_input('Company ID', min_value = 0, max_value = 999, value = 0, help = '0 = Bukan dari Perusahaan')
        days_in_waiting_list = st.number_input('Jumlah Hari di Waiting List', min_value = 0, max_value =999) 
        reserved_room_type = st.selectbox('Jenis Kamar Reservasi', ('A','B','C','D','E','F','G','H','L','P'), help = 'Tidak dicantumkan deskripsi untuk privasi customer')
        assigned_room_type = st.selectbox('Jenis Kamar yang Didapat', ('A','B','C','D','E','F','G','H','I','K','L','P'), help = 'Tidak dicantumkan deskripsi untuk privasi customer')


        st.markdown('---')

        adr = st.number_input('Tarif harian kamar rata-rata ', min_value=0, max_value = 999, value = 100)
        lead_time = st.number_input('Lead Time', min_value=0, max_value=700, value=14, step=1, help='Jarak hari dari mulai reservasi hingga tanggal kedatangan') # Help deskripsi kolom form
        arrival_date_year = st.number_input('Tahun Kedatangan', min_value=2000, max_value=2090, value=2022, help = 'Tahun kedatangan')
        arrival_date_month = st.selectbox('Bulan Kedatangan',('January','February','March','April','May','June','July','August','September','October','November','December'), help = 'Bulan Kedatangan')
        arrival_date_week_number = st.number_input('Minggu Kedatangan',min_value=1, max_value=52, value = 1, help = 'Minggu Kedatangan')
        arrival_date_day_of_month = st.number_input('Hari Kedatangan', min_value=1, max_value=3, value=1, help = 'Hari Kedatangan')
        booking_changes = st.number_input('Jumlah perubahan pada reservasi pada satu transaksi', min_value=0, max_value=99, value=0)
        previous_cancellations = st.number_input('Riwayat Jumlah Reservasi yang Pernah Dibatalkan', min_value=0, max_value=99, value=0)
        previous_bookings_not_canceled = st.number_input('Riwayat Jumlah Reservasi yang tidak Dibatalkan', min_value= 0, max_value = 99, value = 0)
        required_car_parking_spaces = st.number_input('Kebutuhan Lahan Parkir', min_value= 0, max_value=10, value = 0)
        stays_in_weekend_nights = st.number_input('Jumlah Hari Weekend yang direservasi', min_value=0, max_value= 2, value=0)
        stays_in_week_nights = st.number_input('Jumlah Hari Weekday yang direservasi', min_value=0, max_value=5, value=0)       
        total_of_special_requests = st.number_input('Jumlah Permintaan Khusus', min_value=0, max_value=10, value=0)
        
        # Tombol submit
        submitted = st.form_submit_button('Predict')

       

    data_inf = { # Label harus sama dengan CSV, variabel bebas.
        'hotel' : hotel,
        'lead_time' : lead_time,
        'arrival_date_year' : arrival_date_year,
        'arrival_date_month' : arrival_date_month,
        'arrival_date_week_number' : arrival_date_week_number,
        'arrival_date_day_of_month' : arrival_date_day_of_month,
        'stays_in_weekend_nights' : stays_in_weekend_nights,
        'stays_in_week_nights' : stays_in_week_nights,
        'adults' : adults,
        'children' : children,
        'babies' : babies,
        'meal' : meal,
        'country' : country,
        'market_segment' : market_segment,
        'distribution_channel' : distribution_channel,
        'is_repeated_guest' : is_repeated_guest,
        'previous_cancellations' : previous_cancellations,
        'previous_bookings_not_canceled' : previous_bookings_not_canceled,
        'reserved_room_type' : reserved_room_type,
        'assigned_room_type' : assigned_room_type,
        'booking_changes' : booking_changes,
        'deposit_type' : deposit_type,
        'agent' : agent,
        'company' : company,
        'days_in_waiting_list' : days_in_waiting_list,
        'customer_type' : customer_type,
        'adr' : adr,
        'required_car_parking_spaces' : required_car_parking_spaces,
        'total_of_special_requests' : total_of_special_requests,
        'reservation_status' : reservation_status,
        'reservation_status_date' : reservation_status_date

    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        
        # Predict Menggunakan Pipeline
        y_pred_inf = model_pipeline.predict(data_inf)

        if y_pred_inf == 1 :
            st.write('Konsumen akan membatalkan reservasi')
        else :
            st.write('Konsumen tidak akan membatalkan reservasi')

if __name__ == '__main__': # Agar python bisa dibuka standalone tanpa membuka main.
    run()