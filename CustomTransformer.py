import pandas as pd
from sklearn.base import BaseEstimator

# Mendefinisikan Class OutletTypeEncoder
# Class ini akan dijadikan encoder khusus yang digunakan dataset ini dengan temuan EDA penulis
# Encoder ini harus mempunyai fit_transform.
class OutletTypeEncoder(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, df):
        df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
        df['reservation_status_date_year'] = df['reservation_status_date'].dt.year
        df['reservation_status_date_month'] = df['reservation_status_date'].dt.month
        df['reservation_status_date_week'] = df['reservation_status_date'].dt.week
        df['reservation_status_date_day'] = df['reservation_status_date'].dt.day
        df = df.drop('reservation_status_date', axis =1)
        df = df.drop({'agent','company'},axis = 1)
        df['children'].fillna(df['children'].mean(), inplace=True)
        
        return df