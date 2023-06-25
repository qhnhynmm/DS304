import config
from data.data_loader import load_data, split_data
from model.ann_model import train_ann_model, make_ann_prediction
from model.linear_model import train_linear_model, make_linear_prediction

def main():
    # Đọc dữ liệu từ file cấu hình
    data = load_data(config.data_file)

    # Chia dữ liệu thành các cửa sổ với kích thước từ file cấu hình
    windows = split_data(data, config.window_size)

    # Kiểm tra và chạy mô hình ANN nếu được bật trong file cấu hình
    if config.run_ann_model:
        ann_predictions = []
        for window in windows:
            ann_model = train_ann_model(window)
            ann_prediction = make_ann_prediction(ann_model, window[-1])
            ann_predictions.append(ann_prediction)
        # Ghi kết quả vào file đầu ra nếu được cung cấp trong file cấu hình
        if config.output_file:
            predictions_df = pd.DataFrame({'ANN Prediction': ann_predictions})
            predictions_df.to_csv(config.output_file, index=False)

    # Kiểm tra và chạy mô hình Linear Regression nếu được bật trong file cấu hình
    if config.run_linear_model:
        linear_predictions = []
        for window in windows:
            linear_model = train_linear_model(window)
            linear_prediction = make_linear_prediction(linear_model, window[-1])
            linear_predictions.append(linear_prediction)
        # Ghi kết quả vào file đầu ra nếu được cung cấp trong file cấu hình
        if config.output_file:
            predictions_df = pd.DataFrame({'Linear Regression Prediction': linear_predictions})
            predictions_df.to_csv(config.output_file, index=False)

if __name__ == "__main__":
    main()
