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
# dummy variables

dfDummy = pd.get_dummies(df['kota'])
# print(dfDummy)

# ==========================
# join dfDummy to df & drop column 'kota'

dfFinal = pd.concat([df, dfDummy], axis='columns')
dfFinal = dfFinal.drop(['kota'], axis='columns')
# print(dfFinal)

# ==========================
# liner regression

from sklearn.linear_model import LinearRegression
model = LinearRegression()

x = dfFinal.drop(['harga'], axis='columns')
y = dfFinal['harga']

# training
model.fit(x, y)

# slope
print(model.coef_)

# intercept
print(model.intercept_)

# accuracy
print(model.score(x,y))

# ==========================
# prediction

# luas: 2600 di Bekasi, Bogor & Depok
print(model.predict([[2600, 1, 0, 0]]))     # [539709.73984091]
print(model.predict([[2600, 0, 1, 0]]))     # [565396.15136531]
print(model.predict([[2600, 0, 0, 1]]))     # [579723.71533005]