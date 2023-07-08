from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
class DataProcessor:
    def __init__(self):
        self.scaler_MM = MinMaxScaler()
        self.scaler_SD = StandardScaler()
    
    def fill(self, data):
        # Thay các dữ liệu có dạng -999.0 bằng NaN
        data = data.replace({-999.0: np.nan})
        # Backward fill: điền giá trị gần nhất từ phía sau
        data = data.fillna(method='bfill')
        # Forward fill: điền giá trị gần nhất từ phía trước
        data = data.fillna(method='ffill')
        return data
    
    def min_max_scale(self, data):
        features = data.drop(['T2M', 'DATE'], axis=1)
        # Áp dụng MinMax Scaling cho các thuộc tính
        scaled_features = self.scaler_MM.fit_transform(features)
        # Tạo DataFrame mới với các thuộc tính đã scale
        scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
        scaled_data['T2M'] = data['T2M']  # Thêm thuộc tính T2M vào DataFrame đã scale
        scaled_data['DATE'] = data['DATE'] # Thêm thuộc tính DATE vào DataFrame đã scale
        return scaled_data
    
    def standard_scale(self, data):
        features = data.drop(['T2M', 'DATE'], axis=1)
        # Áp dụng Standard Scaling cho các thuộc tính
        scaled_features = self.scaler_SD.fit_transform(features)
        # Tạo DataFrame mới với các thuộc tính đã scale
        scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
        scaled_data['T2M'] = data['T2M']  # Thêm thuộc tính T2M vào DataFrame đã scale
        scaled_data['DATE'] = data['DATE'] # Thêm thuộc tính DATE vào DataFrame đã scale
        return scaled_data
    
    def split_data(self, data):
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X = data.drop(['T2M', 'DATE'], axis=1)
        y = data['T2M']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, y_train, X_test, y_test
