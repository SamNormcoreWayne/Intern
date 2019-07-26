import os
import datetime

import pandas as pd

class parser_csv():
    """
        @attribute: path
        @type: string
        @decription: Store the absolute path of csv file

        @attribute: df
        @type: pandas.DataFrame 
        @description: Store the data

        @method: get_date_n_column(string::column)
        @return: pandas.DataFrame
        @description: 
    """
    def __init__(self, filename):
        self.path = os.path.join(os.getcwd(), 'test_docs',filename)
        self.df = pd.read_csv(self.path).fillna(value=0.0)
        self.df.loc[:, 'Date'] = pd.to_datetime(self.df['Date'],format="%Y-%m-%d")

    def get_date_n_column(self, column):
        return self.df.loc[:, ['Date', column]]
