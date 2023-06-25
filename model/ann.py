import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_ann_model(window):
    # Huấn luyện mô hình ANN
    X = window[:-1]  # Dữ liệu đầu vào
    y = window[-1]  # Dữ liệu đầu ra

    # Chuẩn hóa dữ liệu đầu vào và đầu ra nếu cần
    X_normalized = (X - np.mean(X)) / np.std(X)
    y_normalized = (y - np.mean(y)) / np.std(y)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_normalized, y_normalized, epochs=100, batch_size=32, verbose=0)

    return model

def make_ann_prediction(model, data):
    # Dự đoán dữ liệu với mô hình ANN
    data_normalized = (data - np.mean(data)) / np.std(data)
    prediction_normalized = model.predict(data_normalized)
    prediction = prediction_normalized * np.std(data) + np.mean(data)

    return prediction
