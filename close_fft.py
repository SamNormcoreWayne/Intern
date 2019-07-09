import warnings
import itertools
import pandas as pd
import numpy as np
from pywt import wavedec
from statsmodels import api
from matplotlib import pyplot as plt
from parser_csv import parser_csv
from collections import defaultdict
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

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
    """
    def __init__(self):
        parsed_data = parser_csv("SPY.csv")
        self.name = 'close'
        self.df = parsed_data.get_date_n_column(self.name)

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

    def func_wavelet(self):
        tmp_df = self.df.copy()
        A2, D2, D1 = wavedec(tmp_df[self.name], "db4", level=2)
        # print(type(A2))
        
        plt.figure(figsize=(14, 20))
        plt.plot(A2, label="A2")
        plt.plot(D2, label="D2")
        plt.plot(D1, label="D1")
        plt.legend()
        plt.show()
        
        return [A2, D2, D1]

    def func_acf(self, coeff):
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

    def func_pacf(self, coeff):
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

    def func_diff(self):
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
        """

        #a2_diff[0] = 0
        #d2_diff[0] = 0
        #d1_diff[0] = 0
        """
        plt.figure(figsize=(14, 20))
        plt.plot(a2_diff, label="a2")
        plt.plot(d2_diff, label="d2")
        plt.plot(d1_diff, label="d1")
        plt.legend()
        plt.show()
        """
        
        # a2_diff.dropna(inplace=True)
        # d2_diff.dropna(inplace=True)
        # d1_diff.dropna(inplace=True)
        
        return a2_diff, d2_diff, d1_diff

    def func_ARIMA(self):
        coeff = self.func_wavelet()
        a2 = coeff[0]
        d2 = coeff[1]
        d1 = coeff[2]
        """plt.figure(figsize=(14, 20))
        plt.plot(a2)
        plt.show()"""
        a2_diff, d2_diff, d1_diff = self.func_diff()
        a2_model = ARIMA(a2, order=(1,2,1))
        a2_fit = a2_model.fit(disp=-1)
        #a2_predict = a2_model.predict()
        a2_fitted_value = pd.Series(a2_fit.fittedvalues, copy=True)
        a2_fitted_value_sum = a2_fitted_value.cumsum()
        plt.figure(figsize=(14, 20))
        plt.plot(a2, label="origin")
        plt.plot(a2_diff, label="diff")
        plt.plot(a2_fit.fittedvalues, label="fitted")
        plt.grid()
        plt.legend()
        plt.show()


def main():
    data_day_close = close()
    # data_day_close.func_fft()
    coeff = data_day_close.func_wavelet()
    # print(coeff)
    #data_day_close.func_acf(coeff)
    coeff = list(data_day_close.func_diff())
    # print(coeff)
    data_day_close.func_acf(coeff)
    data_day_close.func_pacf(coeff)
    
    data_day_close.func_ARIMA()

if __name__ == "__main__":
    main()
