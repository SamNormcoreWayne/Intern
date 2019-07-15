import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
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
        self.Row_train, self.Col_train, self.Row_valid, self.Col_valid, self.Row_test, self.Col_test, self.features = self.feature_cls.func_data_split()
    def wavelet(self):
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

    def train_model_origin_interface(self):
        data_train = xgb.DMatrix(self.Row_train, self.Col_train)
        data_valid = xgb.DMatrix(self.Row_valid, self.Col_valid)

        num_rounds = 7000
        watch_lst = [(data_train, 'train'), (data_valid, 'valid')]

        model = xgb.train(self.params, data_train, num_rounds, evals=watch_lst)
        data_test = xgb.DMatrix(self.Row_test)
        predict = model.predict(data_test, ntree_limit=model.best_ntree_limit)
        res = pd.DataFrame(predict, columns=['predict'])
        res['True'] = self.Col_test
        res.to_csv(os.path.join(os.getcwd(), 'res.csv'), index=False)

        score = self.roc_auc_score(self.Col_test, predict)

        xgb.plot_importance(model, max_num_features=-20)
        plt.show()

    def create_feature_map(self):
        with open(os.path.join(os.getcwd(), "xgb.fmap", 'w')) as fp:
            i = 0
            for feat in self.features:
                if feat != '':
                    """
                        This Column should contain some features that do not want to be included
                    """
                    fp.write("{index}\t{feature}\t q \n".format(index=i, feature=feat))
                    i += 1
