import pandas as pd
def load_data(file_name):
  # Đọc dữ liệu từ file CSV
  data = pd.read_csv(file_name)
  return data