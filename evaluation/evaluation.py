import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test):
    r2 = round(r2_score(y_train, y_pred_train),5)
    MSE = round(mean_squared_error(y_train, y_pred_train),5)
    RMSE = round(mean_squared_error(y_train, y_pred_train, squared=False),5)
    MAE = round(mean_absolute_error(y_train, y_pred_train),5)

    print('='*15+'Train'+'='*15)
    print(f'R^2 score: r^2 = {r2}')
    print(f'Mean Squared Error: MSE = {MSE}')
    print(f'Root Mean Squared Error: RMSE = {RMSE}')
    print(f'Mean Absolute Error: MAE = {MAE}')
    MSE = round(mean_squared_error(y_test, y_pred_test),5)
    RMSE = round(mean_squared_error(y_test, y_pred_test, squared=False),5)
    MAE = round(mean_absolute_error(y_test, y_pred_test),5)

    print('='*15+'Test'+'='*15)
    print(f'Mean Squared Error: MSE = {MSE}')
    print(f'Root Mean Squared Error: RMSE = {RMSE}')
    print(f'Mean Absolute Error: MAE = {MAE}')