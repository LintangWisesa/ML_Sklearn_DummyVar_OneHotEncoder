
from sklearn.externals import joblib

model = joblib.load('0e_modeljoblib')

print(model.predict([[89, 21]]))