import os
import warnings
import itertools
import pandas as pd
import numpy as np
import itertools
from prettytable import PrettyTable
from pywt import wavedec, waverec
from statsmodels import api
from matplotlib import pyplot as plt
from parser_csv import parser_csv
from collections import defaultdict
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace import sarimax
from scipy.ndimage.interpolation import shift

"""
Hyper-paramters estimation
Use SARIMAX() in statsmodels to estimate parameters ARIMA(p,d,q)
"""
class ARIMA_estimation():
    def __init__(self, name):
        self.data = parser_csv("SPY.csv")
        self.name = name
        self.df = self.data.get_date_n_column(self.name)
        self.df = pd.DataFrame(self.df[self.name].values, index=self.df['Date'], columns=[self.name])
        p = d = q = range(0, 3)
        self.paramcombo = list(itertools.product(p, d, q))

    def func_wavelet(self):
        tmp_df = self.df.copy()
        A2, D2, D1 = wavedec(tmp_df[self.name], 'db4', mode='sym', level=2)
        return [A2, D2, D1]

    def func_diff(self):
        coeff = self.func_wavelet()
        a2 = coeff[0]
        d2 = coeff[1]
        d1 = coeff[2]

        a2_diff = pd.DataFrame(np.diff(a2, n=1))
        d2_diff = pd.DataFrame(np.diff(d2, n=1))
        d1_diff = pd.DataFrame(np.diff(d1, n=1))

        return [a2_diff, d2_diff, d1_diff]

    def func_estimation(self, coeff):
        combo = self.paramcombo
        table_a2 = PrettyTable(field_names=['a2', 'AIC'])
        table_d2 = PrettyTable(field_names=['d2', 'AIC'])
        table_d1 = PrettyTable(field_names=['d1', 'AIC'])
        a2 = coeff[0]
        d2 = coeff[1]
        d1 = coeff[2]
        for param in combo:
            a2_model = sarimax.SARIMAX(a2, order=param, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
            results_a2 = a2_model.fit()
            d2_model = sarimax.SARIMAX(d2, order=param, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
            results_d2 = d2_model.fit()
            d1_model = sarimax.SARIMAX(d1, order=param, seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
            results_d1 = d1_model.fit()
            # print("ARIMA {} -- AIC: {}".format(param, results.aic))
            table_a2.add_row([param, results_a2.aic])
            table_d2.add_row([param, results_d2.aic])
            table_d1.add_row([param, results_d1.aic])
        print(table_a2.get_string(sortby='AIC'))
        print(table_d2.get_string(sortby='AIC'))
        print(table_d1.get_string(sortby='AIC'))

    def func_ARIMA_model(self, coeff):
        a2 = coeff[0]
        d2 = coeff[1]
        d1 = coeff[2]

        a2_model = ARIMA(a2.values, order=(1, 1, 2))
        d2_model = ARIMA(d2.values, order=(2, 0, 2))
        d1_model = ARIMA(d1.values, order=(1, 0, 2))
        '''
        """
        Ploting
        """
        plt.figure(figsize=(20, 20))
        plt.subplot(221)
        plt.plot(a2, label="origin_a2")
        plt.plot(a2_model.fit(disp=-1).fittedvalues, label="fit_a2")
        plt.legend()
        plt.grid()
        plt.subplot(222)
        plt.plot(d2, label="origin_d2")
        plt.plot(d2_model.fit(disp=-1).fittedvalues, label="fit_d2")
        plt.legend()
        plt.grid()
        plt.subplot(223)
        plt.plot(d1, label="origin_d1")
        plt.plot(d1_model.fit(disp=-1).fittedvalues, label="fit_d1")
        plt.legend()
        plt.grid()
        plt.show()
        """
        Plot End Here
        """
        '''
        a2_fitted = -a2_model.fit(disp=-1).fittedvalues
        d2_fitted = -d2_model.fit(disp=-1).fittedvalues
        d1_fitted = -d1_model.fit(disp=-1).fittedvalues
        return [a2_fitted, d2_fitted, d1_fitted]
    
    def func_waverec(self, coeff, new_coeff):
        a2 = coeff[0]
        d2 = coeff[1]
        d1 = coeff[2]
        a2_new = new_coeff[0]
        d2_new = new_coeff[1]
        d1_new = new_coeff[2]
        denoised = waverec(coeff, 'db4')
        
        """
        Plotting
        """
        plt.figure(figsize=(20, 20))
        plt.plot(denoised)
        plt.show()
        """
        Plot End
        """

def main():
    plt.figure(figsize=(20, 20))
    close = ARIMA_estimation('close')
    coeff = close.func_wavelet()
    a2_b4 = coeff[0]
    d2_b4 = coeff[1]
    d1_b4 = coeff[2]
    plt.subplot(221)
    plt.plot(a2_b4, label='before')
    coeff = close.func_diff()
    #close.func_estimation(coeff)
    coeff = close.func_ARIMA_model(coeff)
    a2 = coeff[0]
    print(a2_b4)
    a2 = np.append([0 , 0], a2)
    a2 = a2_b4 + a2
    """
        此处出错，差分和原数据的相加方式不对，原数据应当后移一位，待修改
    """
    plt.plot(a2, label='after')
    plt.legend()
    plt.subplot(222)
    plt.plot(d2_b4, label='d2_b4')
    d2 = coeff[1]
    d2 = np.append([0], d2)
    d2 = d2_b4 - d2
    plt.plot(d2, label='d2_after')
    plt.legend()
    plt.subplot(223)
    plt.plot(d1_b4, label='d1_b4')
    d1 = coeff[2]
    d1 = np.append([0], d1)
    d1 = d1_b4 - d1
    plt.plot(d1, label='d1_after')
    plt.show()
    #close.func_waverec(coeff)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()