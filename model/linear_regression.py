import numpy as np
from sklearn.linear_model import LinearRegression

def train_linear_model(X,y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def make_linear_prediction(model, data):
    # Dự đoán dữ liệu với mô hình Linear Regression
    prediction = model.predict(data)
    return prediction
