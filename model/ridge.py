import numpy as np
from sklearn.linear_model import Ridge

def train_ridge_model(X,y):
    model = Ridge()
    model.fit(X, y)
    return model
def make_ridge_prediction(model, data):
    # Dự đoán dữ liệu với mô hình Ridge Regression
    prediction = model.predict(data)
    return prediction
