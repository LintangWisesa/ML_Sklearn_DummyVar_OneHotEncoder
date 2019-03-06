import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = [
    {'luas':2600, 'harga':550000, 'kota':'Bekasi'},
    {'luas':3000, 'harga':565000, 'kota':'Bekasi'},
    {'luas':3200, 'harga':610000, 'kota':'Bekasi'},
    {'luas':3600, 'harga':680000, 'kota':'Bekasi'},
    {'luas':4000, 'harga':725000, 'kota':'Bekasi'},
    {'luas':2600, 'harga':585000, 'kota':'Depok'},
    {'luas':2800, 'harga':615000, 'kota':'Depok'},
    {'luas':3300, 'harga':650000, 'kota':'Depok'},
    {'luas':3600, 'harga':710000, 'kota':'Depok'},
    {'luas':2600, 'harga':575000, 'kota':'Bogor'},
    {'luas':2900, 'harga':600000, 'kota':'Bogor'},
    {'luas':3100, 'harga':620000, 'kota':'Bogor'},
    {'luas':3600, 'harga':695000, 'kota':'Bogor'}
]

df = pd.DataFrame(data)
# print(df)

# ==========================
# label encoder with sklearn

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

x = df[['kota', 'luas']].values
y = df['harga']
# print(x)
# print(y)

# ==========================
# one hot encoding with sklearn

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

x = ohe.fit_transform(x).toarray()
print(x)

# Bks, Bgr, Dpk, 2600, 2800, 2900, 3000, 3100, 3200, 3300, 3600, 4000
# [1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]