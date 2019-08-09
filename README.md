# Repo for Intern

**This is an experimental project that uses ARIMA, wavelet and XGBoost to predict stock ups and downs**
**I cannot make sure that the model can be run properly under any circumstances. (Actually even me myself having trouble with debugging...;p)**

Censored some details due to some of them might have secret issues. 

close_fft.py: Considering to change its name in future because the purpose of this file is beyond calculating fft. 

close_ARIMA.py: modelling for ARIMA and wavelet including param estimations, difference, and wavelet trans of time-series.

parser_csv.py: simple version to read csv files and transform data into pandas.DataFrame

XGBoost_stock.py: The XGBoost model for extracting features and predictions. 