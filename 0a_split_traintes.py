import pandas as pd
import numpy as np

dataMobil = [
    {'km':69, 'umur':1, 'harga':199},
    {'km':70, 'umur':2, 'harga':198},
    {'km':71, 'umur':3, 'harga':197},
    {'km':71, 'umur':4, 'harga':196},
    {'km':73, 'umur':5, 'harga':195},
    {'km':74, 'umur':6, 'harga':194},
    {'km':75, 'umur':7, 'harga':193},
    {'km':76, 'umur':8, 'harga':192},
    {'km':77, 'umur':9, 'harga':191},
    {'km':78, 'umur':10, 'harga':190},
    {'km':79, 'umur':11, 'harga':189},
    {'km':80, 'umur':12, 'harga':188},
    {'km':81, 'umur':13, 'harga':187},
    {'km':82, 'umur':14, 'harga':186},
    {'km':83, 'umur':15, 'harga':185},
    {'km':84, 'umur':16, 'harga':184},
    {'km':85, 'umur':17, 'harga':183},
    {'km':86, 'umur':18, 'harga':182},
    {'km':87, 'umur':19, 'harga':181},
    {'km':88, 'umur':20, 'harga':180}
]

mobil = pd.DataFrame(dataMobil)
# print(mobil)

# ===========================

# split datasets = train dataset & test dataset

x = mobil[['km', 'umur']]
y = mobil['harga']

from sklearn.model_selection import train_test_split

# test = 20% & training = 80%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

# ==============================

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x_train, y_train)

print(x_test)
# predict km:89 umur:21
# print(model.predict(x_test))
print(model.predict([[89, 21]]))
# print(y_test)
# print(model.score(x_train, y_train))

# =============================
# save model as file bin => pickle

# import pickle

# with open('0b_modelpickle', 'wb') as modelku:
#     pickle.dump(model, modelku)

# ============================
# save model with joblib

from sklearn.externals import joblib
joblib.dump(model, '0c_modeljoblib')