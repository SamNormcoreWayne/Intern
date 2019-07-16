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
            'booster': 'gbtress',
            'objective': 'multi:softmax',
            'gamma': 0.3,
            'max_depth': 3,
            'alpha': 1e-05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'silent': 1,
            'eta': 0.01,
            'seed': 36,
            'nthread': 4,
            'num_class': 2,
            'scale_pos_weight': 1,
            'n_estimators': 500
        }
        self.low_freq_data
        self.high_freq_data_1
        self.high_freq_data_2
        self.Row_train, self.Col_train, self.Row_valid, self.Col_valid, self.Row_test, self.Col_test, self.features = self.feature_cls.func_data_split()
        self.model

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

        self.model = xgb.train(self.params, data_train, num_rounds, evals=watch_lst)
        data_test = xgb.DMatrix(self.Row_test)
        predict = self.model.predict(data_test, ntree_limit=self.model.best_ntree_limit)
        res = pd.DataFrame(predict, columns=['predict'])
        res['True'] = self.Col_test
        res.to_csv(os.path.join(os.getcwd(), 'res.csv'), index=False)

        score = self.roc_auc_score(self.Col_test, predict)

        xgb.plot_importance(self.model, max_num_features=-20)
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
