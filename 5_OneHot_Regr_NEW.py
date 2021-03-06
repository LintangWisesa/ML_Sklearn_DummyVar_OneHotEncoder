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

dfOnehot = df
dfOnehot['kota'] = lab.fit_transform(dfOnehot['kota'])
# print(dfOnehot)

x = dfOnehot[['kota', 'luas']]
y = dfOnehot['harga']
print(x)
print(y)

# ==========================
# one hot encoding with sklearn

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

z = make_column_transformer(
    (OneHotEncoder(categories='auto'), ['kota']),
    (FunctionTransformer(validate=True), ['luas'])
)

z = z.fit_transform(x)
print(z)

# Bks, Bgr, Dpk, luas
# [0.0e+00 1.0e+00 0.0e+00 3.6e+03]]

# ==========================
# regression

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# training
model.fit(z, y)

# slope
print(model.coef_)

# intercept
print(model.intercept_)

# score
print(model.score(z, y))

# prediction luas 2600 di Bekasi, Bogor, Depok
print(model.predict([[1, 0, 0, 2600]]))
print(model.predict([[0, 1, 0, 2600]]))
print(model.predict([[0, 0, 1, 2600]]))