import config
from data.data_loader import load_data
from data.data_processing import DataProcessor
from model.ann import train_ann_model, make_ann_prediction
from model.linear_regression import train_linear_model, make_linear_prediction
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from model.svm import train_SVM, make_SVM_prediction
import numpy as np
def main():
    # Đọc dữ liệu từ file cấu hình
    data = load_data(config.data_file)
    data_processor = DataProcessor()

    # Sử dụng phương thức preprocess_data
    preprocessed_data, y = data_processor.preprocess_data(data)

    # Sử dụng phương thức split_data
    X_train, y_train, X_test, y_test = data_processor.split_data(preprocessed_data , y)

    # # Kiểm tra và chạy mô hình ANN nếu được bật trong file cấu hình
    # if config.run_ann_model:
    #     ann_predictions = []
    #     for window in windows:
    #         ann_model = train_ann_model(window)
    #         ann_prediction = make_ann_prediction(ann_model, window[-1])
    #         ann_predictions.append(ann_prediction)
    #     # Ghi kết quả vào file đầu ra nếu được cung cấp trong file cấu hình
    #     if config.output_file:
    #         predictions_df = pd.DataFrame({'ANN Prediction': ann_predictions})
    #         predictions_df.to_csv(config.output_file, index=False)
    if config.run_linear_model:  
      linear_model = train_linear_model(X_train,y_train)
      linear_prediction = make_linear_prediction(linear_model, X_test)
      if config.RMSE:
        print('Linear_regression')
        mse = mean_squared_error(y_test,linear_prediction)
        print('MSE:', mse)
    if config.run_SVM_model:  
      SVM_model = train_SVM(X_train,y_train)
      SVM_prediction = make_SVM_prediction(SVM_model, X_test)
      if config.RMSE:
        print('SVM')
        mse = mean_squared_error(y_test,SVM_prediction)
        print('MSE:', mse)
if __name__ == "__main__":
    main()
