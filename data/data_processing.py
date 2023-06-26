from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, data):
        y = data['T2M']
        # Tính toán ma trận tương quan của DataFrame
        corr_matrix = data.corr()
        # Lấy giá trị tuyệt đối của tương quan với cột T2M
        corr_with_T2M = corr_matrix['T2M'].abs()
        # Đặt ngưỡng để xác định mức tương quan thấp
        threshold = 0.1
        # Lọc và xóa các cột có mối tương quan thấp với cột T2M
        columns_to_drop = corr_with_T2M[corr_with_T2M < threshold].index
        X = data.drop(columns=['T2MDEW', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN', 'T2M'])
        X = X.drop(columns=columns_to_drop)
        # Chuẩn hóa dữ liệu về khoảng [0, 1] sử dụng MinMaxScaler
        scaled_data = self.scaler.fit_transform(X)
        return scaled_data, y

    def split_data(self, X, y):
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, y_train, X_test, y_test
