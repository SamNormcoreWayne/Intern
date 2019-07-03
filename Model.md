# Modelling of XGBoost in Stock Prediction

## 1. 模型分析

    1. 股票，时间序列，机器学习，XGBoost
    2. 预测涨幅
    3. 模型假设？
    4. 特征： 
        * 开盘收盘最高最低。其中7和21天移动平均线，指数移动平均线，动量，波林格带，MACD。
        * fft预测长期走势
        * ARIMA预测第二日数据
        * 栈式自动编码（stacked autoencoders， 挖掘在人类语言中无法理解的特征）

## 2. XSBoost简介

    1. 监督学习
    2. CART(Classfication and Regression Tree)
    3. Use the tree to do regression iteratively

### Thoughts

1. fft预测长期趋势。那么如何评价涨跌程度？导数？但是fft是离散的，如何求得导数？用变化率代替？这样的话必定会往当前方向继续增长。
* ~~考虑最后一个点是否为拐点，即导数或变化率是否会为0？~~
* fft拟合后对定义域进行延拓？
2. ARIMA预测短期趋势。数据变大即是增长，变小即是降低。
3. 研究几个股票特征的意义
4. XGBoost本质上实在做数值的回归，那么一般来说，回归分析的曲线不会与原曲线完全重合。利用回归分析得到的曲线来做评估e.g.假设检验 etc。 Because， 回归分析是为了找到数据的统计意义，那么找到统计意义后是否可以利用概率工具进行检验呢？
5. 另外，既然是为了得到t+1天的数据，是否可以对原数据进行数值拟合？ 