import config
from data.data_loader import load_data
from data.data_processing import DataProcessor
from model.linear_regression import train_linear_model, make_linear_prediction
from model.ridge import train_ridge_model, make_ridge_prediction
from model.lstm import train_lstm_model, make_lstm_prediction
from model.cnn1d import train_cnn_model, make_cnn_prediction
from evaluation.evaluation import calculate_metrics
import numpy as np
def main():
    # Đọc dữ liệu từ file cấu hình
    data = load_data(config.data_file)
    data_processor = DataProcessor()
    if config.fill:
      print('\nsử dụng data fill và ')
      data = data_processor.fill(data)
      if config.MinMaxScaler:
        print('MinMaxScaler')
        data = data_processor.min_max_scale(data)
      if config.StandardScaler:
        print('StandardScaler')
        data = data_processor.standard_scale(data)
    else:
      print("\n sử dụng data gốc")
    # chia data
    X_train, y_train, X_test, y_test = data_processor.split_data(data)

    #chạy mô hình Linear Regression nếu được bật trong file cấu hình
    if config.run_linear_model:  
      linear_model = train_linear_model(X_train,y_train)
      y_pred_test = make_linear_prediction(linear_model, X_test)
      y_pred_train = make_linear_prediction(linear_model, X_train)
      print("\nlinear_regression")
      calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
      print('\n\n')
    #chạy mô hình Ridge Regression nếu được bật trong file cấu hình
    if config.run_ridge_model:  
      ridge_model = train_ridge_model(X_train,y_train)
      y_pred_test = make_ridge_prediction(ridge_model, X_test)
      y_pred_train = make_ridge_prediction(ridge_model, X_train)
      print("ridge_regression")
      calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
      print('\n\n')
    #chạy mô hình lstm nếu được bật trong file cấu hình
    if config.run_lstm_model:  
      lstm_model, X_train_lstm, X_test_lstm = train_lstm_model(X_train,X_test,y_train)
      y_pred_test = make_lstm_prediction(lstm_model, X_test_lstm)
      y_pred_train = make_lstm_prediction(lstm_model,X_train_lstm)
      print("\nlstm_regression")
      calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
      print('\n\n')
    #chạy mô hình lstm nếu được bật trong file cấu hình
    if config.run_cnn_model:  
      cnn_model, X_train_cnn, X_test_cnn = train_cnn_model(X_train,X_test,y_train)
      y_pred_test = make_cnn_prediction(cnn_model, X_test_cnn)
      y_pred_train = make_cnn_prediction(cnn_model,X_train_cnn)
      print("\ncnn1d_regression")
      calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
      print('\n\n')
    
    
if __name__ == "__main__":
    main()
