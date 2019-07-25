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
    """
        @attribute: data
        @type: pandas.DataFrame

        @attribute: name
        @type: str

        @attribute: df
        @type: pandas.DataFrame

        @attribute: paramcombo
        @type: list<tuple>[27]

        @method: func_wavelet
        @param: self
        @return: list<pd.Series>[3]
        @desc: use pywt.wavedec with 'db4' to decomp data

        @method: func_diff
        @param: self
        @return: list<pd.Series>[3]
        @desc: do difference to data

        @method: func_estimation
        @param: list<pd.Series> *coeff
        @return: None
        @desc: param estimation of ARIMA

        @method: func_ARIMA_model
        @param: list<pd.Series> *coeff
        @return: list<pd.Series>[3]
        @desc: the ARIMA model of three layers data from wavelet //// do not ask me why there is a minus

        @method: func_waverec
        @param: list<pd.Series> *coeff
        @return: None
        @desc: pywt.waverec inverse dwt. input data into csv file. 
    """
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
        return [pd.Series(A2), pd.Series(D2), pd.Series(D1)]

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

        a2_model = ARIMA(a2.values, order=(1, 2, 2))
        d2_model = ARIMA(d2.values, order=(2, 0, 2))
        d1_model = ARIMA(d1.values, order=(0, 0, 2))
        a2_model_fit = a2_model.fit(disp=-1)
        d2_model_fit = d2_model.fit(disp=-1)
        d1_model_fit = d1_model.fit(disp=-1)
        '''
        """
        Ploting
        """
        plt.figure(figsize=(20, 20))
        plt.subplot(221)
        plt.plot(a2, label="origin_a2")
        plt.plot(-a2_model_fit.fittedvalues, label="fit_a2")
        plt.legend()
        plt.grid()
        plt.subplot(222)
        plt.plot(d2, label="origin_d2")
        plt.plot(d2_model_fit.fittedvalues, label="fit_d2")
        plt.legend()
        plt.grid()
        plt.subplot(223)
        plt.plot(d1, label="origin_d1")
        plt.plot(d1_model_fit.fittedvalues, label="fit_d1")
        plt.legend()
        plt.grid()
        plt.show()
        """
        Plot End Here
        """
        '''
        a2_fitted = -a2_model_fit.fittedvalues
        d2_fitted = d2_model_fit.fittedvalues
        d1_fitted = d1_model_fit.fittedvalues
        return [a2_fitted, d2_fitted, d1_fitted]
    
    def func_waverec(self, coeff, new_coeff):
        a2 = coeff[0]
        d2 = coeff[1]
        d1 = coeff[2]
        a2_new = new_coeff[0]
        d2_new = new_coeff[1]
        d1_new = new_coeff[2]
        a2_new = np.append([a2[0], 0], a2_new)
        a2_new = shift(a2, 1) + a2_new
        d2_new = shift(d2, -1) + d2_new
        d1_new = shift(d1, -1) + d1_new
        """
            Use scipy.ndimage.interpolation.shift to shift data in np.ndarray
            Or use pd.DataFrame.shift()
            Add the shifted data to the diffed data.
        """
        denoised = waverec([a2_new, d2_new, d1_new], 'db4')
        """
            Plotting
        """
        plt.figure(figsize=(20, 20))
        plt.plot(denoised, label='wavelet')
        plt.plot(self.df['close'].values, label='origin')
        plt.legend()
        plt.show()
        """
        Plot End
        """
        self.data.df['denoised'] = denoised[1:]
        self.data.df.to_csv('denoised.csv', index=False)

def main():
    plt.figure(figsize=(14, 14))
    close = ARIMA_estimation('close')
    coeff = close.func_wavelet()
    #close.func_estimation(coeff)
    new_coeff = close.func_ARIMA_model(coeff)

    a2_b4 = coeff[0]
    d2_b4 = coeff[1]
    d1_b4 = coeff[2]
    # plt.subplot(221)
    plt.plot(a2_b4, label='before')
    #close.func_estimation(coeff)
    a2 = new_coeff[0]
    print(a2_b4)
    a2 = np.append([a2_b4[0], 0], a2)
    a2 = shift(a2_b4, 1) + a2
    plt.plot(a2, label='after')
    plt.legend()
    plt.figure(figsize=(14, 14))
    # plt.subplot(222)
    plt.plot(d2_b4, label='d2_b4')
    d2 = new_coeff[1]
    d2 = shift(d2_b4, -1) + d2
    plt.plot(d2, label='d2_after')
    plt.legend()
    #plt.subplot(223)
    plt.figure(figsize=(14, 14))
    plt.plot(d1_b4, label='d1_b4')
    d1 = new_coeff[2]
    d1 = shift(d1_b4, -1) + d1
    plt.plot(d1, label='d1_after')
    plt.show()

    close.func_waverec(coeff, new_coeff)
    # close.test()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()