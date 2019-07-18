import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import operator
from close_fft import close
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error


class close_xgboost():
    """
    Static variables for close_xgboost
    """
    
    def __init__(self, filename):
        self.filename = filename
        self.feature_cls = close(self.filename)
        self.params = {
            """
                Parameters here.
            """
        }
        self.low_freq_data
        self.high_freq_data_1
        self.high_freq_data_2
        # self.Row_train, self.Col_train, self.Row_valid, self.Col_valid, self.Row_test, self.Col_test, self.features = self.feature_cls.func_data_split()
        self.model

    def wavelet(self):
        """
            @varName: coeff
            @varType: list<numpy.ndarray>[3]
        """
        coeff = self.feature_cls.func_wavelet()
        self.low_freq_data = coeff[0]
        self.high_freq_data_1 = coeff[1]
        self.high_freq_data_2 = coeff[2]

    def roc_auc_score(self, x, y):
        return metrics.roc_auc_score(x, y)

    def func_auc(self, y, predict):
        fpr, tpr, threshold = metrics.roc_curve(y, predict, pos_label=2)
        return metrics.auc(fpr, tpr)

    def mean_abs_err(self, y_data, train_data):
        y = train_data.get_label()
        """
        the train_data shall be pandas.Series/Dataframe
        """
        return mean_absolute_error(np.exp(y), np.exp(y_data))

    '''
    def train_model_origin_interface(self):
        data_train = xgb.DMatrix(self.Row_train, self.Col_train)
        data_valid = xgb.DMatrix(self.Row_valid, self.Col_valid)

        num_rounds = 7000
        watch_lst = [(data_train, 'train'), (data_valid, 'valid')]

        self.model = xgb.train(self.params, data_train, num_rounds, evals=watch_lst)
        data_test = xgb.DMatrix(self.Row_test)
        predict = self.model.predict(data_test, ntree_limit=self.model.best_ntree_limit)
        res = pd.DataFrame(predict, columns=['predict'])
        res['True'] = self.Col_test
        res.to_csv(os.path.join(os.getcwd(), 'res.csv'), index=False)

        score = self.roc_auc_score(self.Col_test, predict)

        xgb.plot_importance(self.model, max_num_features=-20)
        plt.show()
    '''
    def split_data(self, data):
        """
            @param: data
            @type: numpy.ndarray[n, 1]

            @return: d_train, d_cv, d_test
            @type: list<numpy.ndarray>[3]
        """
        test_size = 0.045
        cv_size = 0.2
        train_size = 1 - cv_size - test_size
        length = data.shape[0]
        
        test_size *= length
        cv_size *= length
        train_size *= length

        d_train = data[:train_size]
        d_cv = data[train_size : cv_size]
        d_test = data[cv_size : ]
        return [d_train, d_cv, d_test]

    def train_model_with_wavelet(self):
        low_freq_set = self.split_data(self.low_freq_data)
        high_freq_1_set = self.split_data(self.high_freq_data_1)
        high_freq_2_set = self.split_data(self.high_freq_data_2)

        low_freq_coeff_train = low_freq_set[0]
        low_freq_coeff_cv = low_freq_set[1]
        low_freq_coeff_test = low_freq_set[2]

        high_freq_data_1_train = high_freq_1_set[0]
        high_freq_data_1_cv = high_freq_1_set[1]
        high_freq_data_1_test = high_freq_1_set[2]

        high_freq_data_2_train = high_freq_2_set[0]
        high_freq_data_2_cv = high_freq_2_set[1]
        high_freq_data_2_test = high_freq_2_set[2]

    """
    def create_feature_map(self):
        with open(os.path.join(os.getcwd(), "xgb.fmap", 'w')) as fp:
            i = 0
            for feat in self.features:
                if feat != '':
                    '''
                        This Column should contain some features that do not want to be included
                    '''
                    fp.write("{index}\t{feature}\t q \n".format(index=i, feature=feat))
                i += 1
    """
    def feature_importance(self):
        importance = self.model.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        print(df[df['fscore'] < 0.001]['feature'].tolist())

def main():
    model = close_xgboost("SPY.csv")
    model.wavelet()
    model.train_model_origin_interface()
    model.create_feature_map()
