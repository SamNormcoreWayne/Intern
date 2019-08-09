import os
import talib
import warnings
import itertools
import numpy as np
import pandas as pd
from pywt import wavedec
from talib import MA_Type
from statsmodels import api
from parser_csv import parser_csv
from collections import defaultdict
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class close():
    """
        @attribute: df
        @type: pandas.dataframe

        @method: func_fft
        @param: self
        @return: None
        @description: fft

        @method: func_wavelet
        @param: self
        @return: list<numpy.ndarray>
        @description: wavelet stock series into 1 high freq and 2 low freq

        @method: func_acf
        @param: self, list<numpy.ndarray> coeff
        @return: tuple<numpy.ndarray>[3]
        @description: receive the results of func_wavelet and return the acf.

        @method: func_pacf
        @description: similar with func_acf

        @method: func_ARIMA
        @param: self
        @return: None
        @description: ARIMA Model

        @method: fun_data_split
        @param: self
        @return: 
    """
    def __init__(self, filename : str):
        self.filename = filename
        parsed_data = parser_csv(self.filename)
        self.name = 'close'
        self.df = parsed_data.df
        # self.df.loc[:, 'Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d')

    def func_fft(self):
        tmp_df = self.df.copy()
        vec = np.asarray(tmp_df[self.name].tolist())
        vec_fft = np.fft.fft(vec)
        fft_pd_series = pd.Series(vec_fft)
        tmp_df[self.name + '_abs'] = fft_pd_series.apply(lambda x : np.abs(x))
        tmp_df[self.name + '_angle'] = fft_pd_series.apply(lambda x : np.angle(x))
        # tmp_df[self.name + 'fft'] = vec_fft
        #ax = plt.gca()

        """
            Change the scale of axis
        """
        #ax.xaxis.set_major_locator(plt.MultipleLocator(50))
        #ax.yaxis.set_major_locator(plt.MultipleLocator(100))
        plt.figure(figsize=(14, 7), dpi=100)
        plt.plot(vec_fft, label='origin')
        plt.figure(figsize=(14, 7), dpi=100)
        for num in [6, 9, 100, 200, 500]:
            fft_copy = np.asarray(vec_fft.tolist()); fft_copy[num : -num] = 0
            #plt.plot(fft_copy, label='fourier {}'.format(num))
            plt.plot(np.fft.ifft(fft_copy), label='{}'.format(num))
            '''
                Inverse FFT to get the curve fitting
            '''
            """
                Gibbs phenomenon happens between ([6660, 6670] days, [6500, 6550] USD)
            """
        plt.plot(tmp_df[self.name], label='Origin', marker='x')
        plt.grid()
        plt.legend()
        """
        plt.figure(figsize=(14, 7), dpi=100)
        plt.plot(tmp_df['Date'], tmp_df[self.name + '_abs'])
        plt.figure(figsize=(14, 7), dpi=100)
        plt.plot(tmp_df['Date'], tmp_df[self.name + 'fft'])
        plt.figure(figsize=(15, 8), dpi=100)
        plt.plot(tmp_df['Date'], tmp_df[self.name + '_angle'])
        """
        plt.show()

    def func_wavelet(self) -> [list]:
        tmp_df = self.df.copy()
        A2, D2, D1 = wavedec(tmp_df[self.name], "db4", level=2)
        # print(type(A2))
        #A2 = pd.DataFrame(A2, index)
        """
        plt.figure(figsize=(14, 20))
        plt.plot(A2, label="A2")
        plt.plot(D2, label="D2")
        plt.plot(D1, label="D1")
        plt.legend()
        plt.show()
        """
        return [A2, D2, D1]

    def func_acf(self, coeff : [pd.DataFrame]) -> (pd.DataFrame):
        """
            @param: list<np.narray> coeff
        """
        # print(type(np.asarray(coeff[0])))
        a2_acf, a2_confi = acf(np.asarray(coeff[0]), nlags=20, alpha=True)
        d2_acf, d2_confi = acf(np.asarray(coeff[1]), nlags=20, alpha=True)
        d1_acf, d1_confi = acf(np.asarray(coeff[2]), nlags=20, alpha=True)

        """
        Draw the images of a2, d2, d1 acf
        """
        plt.figure(figsize=(14, 20))
        plt.subplot(221)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.plot(np.asarray(a2_acf))
        plt.title("A2")
        plt.grid()
        plt.axhline(y=0, linestyle="--", color="red")
        plt.axhline(y=-1.96/np.sqrt(len(a2_acf)), linestyle="--", color="red")
        plt.axhline(y=1.96/np.sqrt(len(a2_acf)), linestyle="--", color="red")
        plt.subplot(222)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.plot(np.asarray(d2_acf))
        plt.title("D2")
        plt.grid()
        plt.axhline(y=0, linestyle="--", color="red")
        plt.axhline(y=-1.96/np.sqrt(len(d2_acf)), linestyle="--", color="red")
        plt.axhline(y=1.96/np.sqrt(len(d2_acf)), linestyle="--", color="red")
        plt.subplot(223)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.plot(np.asarray(d1_acf))
        plt.title("D1")
        plt.grid()
        plt.axhline(y=0, linestyle="--", color="red")
        plt.axhline(y=-1.96/np.sqrt(len(d1_acf)), linestyle="--", color="red")
        plt.axhline(y=1.96/np.sqrt(len(d1_acf)), linestyle="--", color="red")
        plt.show()
        """
        Draw end
        """
        return a2_acf, d2_acf, d1_acf

    def func_pacf(self, coeff : [pd.DataFrame]):
        a2_acf, a2_confi = pacf(np.asarray(coeff[0]), nlags=20, method='ols', alpha=True)
        d2_acf, d2_confi = pacf(np.asarray(coeff[1]), nlags=20, method='ols', alpha=True)
        d1_acf, d1_confi = pacf(np.asarray(coeff[2]), nlags=20, method='ols', alpha=True)

        plt.figure(figsize=(14, 20))
        plt.subplot(221)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.plot(np.asarray(a2_acf))
        plt.title("A2")
        plt.grid()
        plt.axhline(y=0, linestyle="--", color="red")
        plt.axhline(y=-1.96/np.sqrt(len(a2_acf)), linestyle="--", color="red")
        plt.axhline(y=1.96/np.sqrt(len(a2_acf)), linestyle="--", color="red")
        plt.subplot(222)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.plot(np.asarray(d2_acf))
        plt.title("D2")
        plt.grid()
        plt.axhline(y=0, linestyle="--", color="red")
        plt.axhline(y=-1.96/np.sqrt(len(d2_acf)), linestyle="--", color="red")
        plt.axhline(y=1.96/np.sqrt(len(d2_acf)), linestyle="--", color="red")
        plt.subplot(223)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.plot(np.asarray(d1_acf))
        plt.title("D1")
        plt.grid()
        plt.axhline(y=0, linestyle="--", color="red")
        plt.axhline(y=-1.96/np.sqrt(len(d1_acf)), linestyle="--", color="red")
        plt.axhline(y=1.96/np.sqrt(len(d1_acf)), linestyle="--", color="red")
        plt.show()

    def func_diff(self) -> (np.ndarray):
        """
            @returnType: tuple<pd.Series>
        """
        coeff = self.func_wavelet()
        a2, d2, d1 = coeff[0], coeff[1], coeff[2]
        # a2_diff = pd.Series(a2).diff(1)
        # d2_diff = pd.Series(d2).diff(1)
        # d1_diff = pd.Series(d1).diff(1)
        a2_diff = np.diff(a2, n=1)
        d2_diff = np.diff(d2, n=1)
        d1_diff = np.diff(d1, n=1)
        """
            After diff, we will have several Series where have 'NaN' vaules in loc(0) of each Series
            This outcome causes some errors during the procedure of acf.
            Use numpy.diff instead of pd.Series.diff to solve this problem.
        """

        #a2_diff[0] = 0
        #d2_diff[0] = 0
        #d1_diff[0] = 0

        """"""
        """plt.figure(figsize=(14, 20))
        plt.plot(a2_diff, label="a2")
        plt.plot(d2_diff, label="d2")
        plt.plot(d1_diff, label="d1")
        plt.legend()
        plt.show()"""

        # a2_diff.dropna(inplace=True)
        # d2_diff.dropna(inplace=True)
        # d1_diff.dropna(inplace=True)

        return a2_diff, d2_diff, d1_diff

    def func_ARIMA(self) -> [np.ndarray]:
        """
        parameters in ARIMA should follow the results of experiment in close_ARIMA.py
        And, we won't use the diff function: 'self.func_diff'
        Because on the basis of 'close_ARIMA.py'
        ARIMA(2, 1, 2) deals with the low freq data well.
        HOWEVER, we would like to do more exp in future by diff funcs and ARMA due to the limitaion
        in ARIMA that we can only do differece twice. 
        """
        coeff = self.func_wavelet()
        a2 = coeff[0]
        d2 = coeff[1]
        d1 = coeff[2]
        """plt.figure(figsize=(14, 20))
        plt.plot(a2)
        plt.show()"""
        a2_diff, d2_diff, d1_diff = self.func_diff()
        a2_model = ARIMA(a2, order=(1,2,2))
        d2_model = ARIMA(d2, order=(2,0,2))
        d1_model = ARIMA(d1, order=(0,0,2))
        a2_fit = a2_model.fit(disp=-1)
        d2_fit = d2_model.fit(disp=-1)
        d1_fit = d1_model.fit(disp=-1)
        '''
        plt.figure(figsize=(40, 40))
        #plt.plot(a2, label="origin")
        plt.subplot(221)
        plt.plot(a2_diff, label="diff")
        plt.plot(a2_fit.fittedvalues, label="fitted")
        plt.grid()
        plt.legend()
        plt.subplot(222)
        plt.plot(d2_diff, label="diff_d2")
        plt.plot(d2_fit.fittedvalues, label="fitted_d2")
        plt.grid()
        plt.legend()
        plt.subplot(223)
        plt.plot(d1_diff, label="diff_d1")
        plt.plot(d1_fit.fittedvalues, label="fitted_d1")
        plt.grid()
        plt.legend()
        plt.show()
        '''
        a2_fitted = -a2_fit.fittedvalues
        d2_fitted = d2_fit.fittedvalues
        d1_fitted = d1_fit.fittedvalues
        return [a2_fitted, d2_fitted, d1_fitted]

    def func_standarized(self, data : np.ndarray) -> "sklearn.preprocessing.StandardScaler().fit_transform() -> np.ndarray":
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def func_norm(self, data : np.ndarray) -> "sklearn.preprocessing.MinMaxScaler().transform() -> np.ndarray":
        scaler = MinMaxScaler()
        return scaler.transform(data)

    def time_series_to_supervised(self, length_in, length_out, drop=True):
        """
        Split train set into several sets.
        One dimension into mulit-dimension?
        """
        columns = self.df.shape[1]

        df = self.df
        cols, names = list(), list()
        for i in range(length_in, 0, -1):
            cols.append(df.shift(i))
            if i == 0:
                names += [('var{}(t)'.format(j + 1)) for j in range(columns)]
            else:
                names += [('var{}(t + {})'.format(j + 1, i) for j in range(columns))]
        data = pd.concat(cols, axis=1)
        data.columns = names
        if drop:
            data.dropna(inplace=True)
        return data

    """
        Lacking of one function to get some tech indicator
        Is this why causes the error in xgb training?
    """
    def tech_indicator(self) -> pd.DataFrame:
        cp = self.df.copy()
        close = cp.close.values
        open_price = cp.close.values
        high = cp.close.values
        low = cp.low.values
        volume = cp.volume.values

        cp['SMA_5'] = talib.SMA(close, 5)
        cp['upper'], cp['middle'], cp['low'] = talib.BBANDS(close, matype=MA_Type.T3)
        cp['slowk'], cp['slowd'] = talib.STOCH(
            high,
            low,
            close,
            fastk_period=9,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        cp['slowk_slowd'] = cp['slowk'] - cp['slowd']
        cp['CCI'] = talib.CCI(high, low, close, timeperoid=10)
        cp['DIF'], cp['DEA'], cp['HIST'] = talib.MACD(
            close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        cp['BAR'] = (cp['DIF'] - cp['DEA']) * 2

        cp['macdext'], cp['signal_xt'], cp['hist_xt'] = talib.MACDFIX(close, signalperiod=9)

        cp['RSI'] = talib.RSI(close, timepriod=14)
        cp['AROON_DOWN'], cp['AROON_UP'] = talib.AROON(high, low, timeperiod=14)

        cp['upperband'], cp['middleband'], cp['lowerband'] = talib.BBANDS(
            close,
            timepriod=5,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )

        cp['AD'] = talib.AD(high, low, close, volume)
        cp['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        cp['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
        cp['OBV'] = talib.OBV(close, volume)

        return cp
    def func_data_split(self) -> "tuple(pd.Dataframe)[7]":
        tmp_df = self.df
        test_size = 0.045
        cv_size = 0.2

        target = 'close'
        drop_col_0 = ['Date', target]
        drop_col_1 = []
        drop_col = drop_col_0 + drop_col_1

        num_valid = int(len(tmp_df) * cv_size)
        num_test = int(len(tmp_df) * test_size)
        num_train = len(tmp_df) - num_valid - num_test

        train_data = tmp_df[:num_train]
        valid_data = tmp_df[num_train:num_train + num_valid]
        train_valid_data = tmp_df[:num_train + num_valid]
        test_data = tmp_df[num_train + num_valid:]

        train_valid_data.to_csv(os.path.join(os.getcwd(), 'test_docs', 'train.csv'), index=False)
        test_data.to_csv(os.path.join(os.getcwd(), 'test_docs', 'test.csv'), index=False)

        Col_train = train_data[target].values
        Col_valid = valid_data[target].values
        Col_test = test_data[target].values

        Row_train = train_data.drop(drop_col, axis=1)
        Row_valid = valid_data.drop(drop_col, axis=1)
        Row_test = test_data.drop(drop_col, axis=1)
        features = test_data.columns.values.tolist()
        Row_train = self.func_standarized(Row_train.values)
        Row_valid = self.func_standarized(Row_valid.values)
        Row_test = self.func_standarized(Row_test.values)

        return Row_train, Col_train, Row_valid, Col_valid, Row_test, Col_test, features

    def func_data_split_with_param(self, data):
        """
            @param: data
            @type: numpy.ndarray
        """

    def func_one_hot(self, data):
        if data > 0:
            new = 1
        else:
            new = 0
        return new


def main():
    data_day_close = close("denoised.csv")
    # data_day_close.func_fft()
    #coeff = data_day_close.func_wavelet()
    # print(coeff)
    #print(type(coeff[0]))
    #data_day_close.func_acf(coeff)
    #coeff = list(data_day_close.func_diff())
    # print(coeff)
    #data_day_close.func_acf(coeff)
    #data_day_close.func_pacf(coeff)

    data_day_close.func_ARIMA()


if __name__ == "__main__":
    main()
