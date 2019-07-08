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
        @varNmae: df
        @varType: pandas.dataframe
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
        return [A2, D2, D1]

    def func_acf(self):
        coeff = self.func_wavelet()
        a2_acf, a2_confi = acf(coeff[0], nlags=20, alpha=True)
        d2_acf, d2_confi = acf(coeff[1], nlags=20, alpha=True)
        d1_acf, d1_confi = acf(coeff[2], nlags=20, alpha=True)
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

    def func_pacf(self):
        None


def main():
    data_day_close = close()
    # data_day_close.func_fft()
    data_day_close.func_acf()


if __name__ == "__main__":
    main()
