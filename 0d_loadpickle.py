import pickle

with open('0c_modelpickle', 'rb') as modelku:
    model = pickle.load(modelku)

print(model.predict([[90, 22]]))