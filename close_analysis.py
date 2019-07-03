import warnings
import itertools
import pandas as pd
import numpy as np
from statsmodels import api
from matplotlib import pyplot as plt
from parser_csv import parser_csv
from collections import defaultdict

class close():
    """
        @varNmae: df
        @varType: pandas.dataframe
    """
    def __init__(self):
        parsed_data = parser_csv()
        self.name = 'close'
        self.df = parsed_data.get_date_n_column(self.name)

    def func_fft(self):
        tmp_df = self.df.copy()
        vec = np.asarray(tmp_df[self.name].tolist())
        vec_fft = np.fft.fft(vec)
        fft_pd_series = pd.Series(vec_fft)
        tmp_df[self.name + '_abs'] = fft_pd_series.apply(lambda x : np.abs(x))
        tmp_df[self.name + '_angle'] = fft_pd_series.apply(lambda x : np.angle(x))
        tmp_df[self.name + 'fft'] = vec_fft
        plt.figure(figsize=(14, 7), dpi=100)
        for num in [3, 6, 9, 100]:
            fft_copy = np.asarray(vec_fft.tolist())
            fft_copy[num : -num] = 0
            plt.plot(np.fft.ifft(fft_copy), label='{}'.format(num))
        plt.plot(tmp_df[self.name], label='Origin')
        plt.legend()
        plt.show()


def main():
    data_day_close = close()
    data_day_close.func_fft()


if __name__ == "__main__":
    main()
