import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

def train_lstm_model(X_train, X_test, y_train):
    # Reshape dữ liệu để phù hợp với mạng LSTM
    X_train_lstm = X_train.values
    X_test_lstm = X_test.values
    n = len(X_train.columns)
    X_train_lstm = X_train_lstm.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test_lstm.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Xây dựng mô hình LSTM
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, n)))
    model.add(Dense(1))

    # Tạo checkpoint để lưu best model
    checkpoint = ModelCheckpoint('lstm.h5', monitor='loss', save_best_only=True, mode='min')

    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='mse')


    # Huấn luyện mô hình
    model.fit(X_train_lstm, y_train, epochs=150, batch_size=32, verbose=1, callbacks=[checkpoint])

    lstm_model = load_model('lstm.h5')
    return lstm_model,X_train_lstm,X_test_lstm

def make_lstm_prediction(model, data):
    # Dự đoán dữ liệu với mô hình Linear Regression
    prediction = model.predict(data)
    return prediction
