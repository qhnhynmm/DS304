import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

def train_cnn_model(X_train, X_test, y_train):
    X_train_cnn = X_train.values
    X_test_cnn = X_test.values
    n = len(X_train.columns)
    # Reshape lại dữ liệu đầu vào để phù hợp với kiến trúc CNN
    X_train_cnn = X_train_cnn.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test_cnn.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Xây dựng mô hình CNN
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n, 1)))
    cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dense(1, activation='linear'))

    # Tạo checkpoint để lưu best model
    checkpoint = ModelCheckpoint('cnn.h5', monitor='loss', save_best_only=True, mode='min')

    # Biên dịch mô hình
    cnn_model.compile(loss='mean_squared_error', optimizer='adam')

    # Huấn luyện mô hình
    cnn_model.fit(X_train_cnn, y_train, epochs=150, batch_size=64, callbacks=[checkpoint])

    cnn_model = load_model("cnn.h5")
    return cnn_model,X_train_cnn,X_test_cnn

def make_cnn_prediction(model, data):
    # Dự đoán dữ liệu với mô hình Linear Regression
    prediction = model.predict(data)
    return prediction
