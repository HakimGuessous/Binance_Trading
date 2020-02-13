import pandas as pd
import numpy as np
import os
import datetime as dt
import random
import sklearn as sk
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
plt.interactive(False)
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn.metrics as skll
import talib as ta
import joblib
from sklearn.preprocessing import MinMaxScaler


#Set working directory
os.chdir('C:/Users/HGuessous/Documents/Hakim Stuff/Binance_Trading')

#Read minute level BTC to USD data (which can be downloaded from the Binance API)
btc = pd.read_csv('btcusd.csv')
#Clean dataframe and change to 10 minute candles
btc['date'] = pd.to_datetime(btc['time'], unit='ms')
btc = btc.set_index('date')
btc = btc.loc[:, ~btc.columns.str.contains('^Unnamed')]
btc['duplicate'] = np.sum(btc == btc.shift(periods=1), axis=1)
btc = btc[btc['duplicate'] < 5]
del btc['duplicate']
del btc['time']

btc = (btc.resample('10T')
     .agg({'open': 'first', 'close': 'last',
           'high': np.max, 'low': np.min,
           'volume': np.sum}))
btc = btc.dropna()


#Function that finds local minimum and maximums in the price history
def get_max_min(prices, smoothing, window_range, order):
    smooth_prices = prices['close'].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater, order=order)[0] + smoothing
    local_min = argrelextrema(smooth_prices.values, np.less, order=order)[0] + smoothing
    price_local_max_dt = []
    for i in local_max:
        if (i > window_range) and (i < len(prices) - smoothing):
            price_local_max_dt.append(prices.iloc[i - window_range:i ]['close'].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i > window_range) and (i < len(prices) - smoothing):
            price_local_min_dt.append(prices.iloc[i - window_range:i ]['close'].idxmin())
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min = max_min.reset_index()
    max_min = max_min[~max_min.duplicated()]
    p = prices.reset_index()
    max_min['min_num'] = p[p['date'].isin(max_min.date)].index.values
    max_min = max_min.set_index('min_num').loc[:,['close','avg_vol']]

    return max_min


#Creates dataframe where each row contains the last n minimums/maximums as a % of current price, and the associated
# timeframe/volume. The purpose is to simplify recent price movements without loosing information.

def get_trend(prices, minmax, order, steps):
    minmax1 = minmax
    for i in range(1,steps+1):
        minmax1['close'+ str(i)] = minmax1['close'].shift(periods=i)

    minmax1['index0'] = minmax1.index
    for i in range(1,steps+1):
        minmax1['index'+ str(i)] = minmax1['index0'].shift(periods=i)

    for i in range(1,steps):
        minmax1['avg_vol'+ str(i+1)] = minmax1['avg_vol'].shift(periods=i)

    pricesdata = pd.merge(prices, minmax1, how='left', left_index=True, right_index=True)
    pricesdata.iloc[:,6:len(pricesdata.columns)] = pricesdata.iloc[:,6:len(pricesdata.columns)].fillna(method='ffill').shift(periods=order)
    pricesdata=pricesdata.rename(columns={"close_x":"close","close_y":"close0","avg_vol_y":"avg_vol1","avg_vol_x":"avg_vol0"})

    for i in range(0,steps+1):
        pricesdata['close'+ str(i)] = ((pricesdata['close']/pricesdata['close'+ str(i)])*100)-100

    for i in range(0,steps+1):
        pricesdata['index'+ str(i)] = pricesdata.index - pricesdata['index'+ str(i)]

    for i in range(1,steps+1):
        pricesdata['avg_vol'+ str(i)] = ((pricesdata['avg_vol'+ str(i)] / pricesdata['avg_vol0'])*100)-100

    pricesdata = pricesdata.dropna()
    pricesdata = pricesdata.drop('avg_vol0',1)
    return pricesdata

#Adds the future price and whether BTC went up or down from the current price
def get_data(prices, smoothing, window_range, order, steps):
    minmax = get_max_min(prices, smoothing, window_range, order)
    prices = prices.reset_index()
    prices['future'] = prices['close'].shift(periods=-12)
    prices['future_b'] = np.where(prices['future'] > prices['close'], 1, 0)

    prices1 = get_trend(prices, minmax, order, steps)

    return prices1


#Function settings for price smoothing and number of minimums/maximums to include as features
order = 5
window_range = order
smoothing = 3
steps = 10
prices = btc.loc[dt.date(2013,4,27):dt.date(2019,6,28)]
prices = prices.drop(['open','high','low'],axis = 1)
prices['avg_vol'] = prices['volume'].rolling(window=int(order/2)).mean()

prices1 = get_data(prices, smoothing, window_range, order, steps)
prices1 = prices1.set_index(pd.DatetimeIndex(prices1['date']))


#Plot a single random row of the output dataframe as a reference

#test_prices = prices1
#counter = random.choice(test_prices.index)
#test = np.array([counter-test_prices.loc[counter,'index0_x':('index'+str(steps))+'_x'],test_prices.loc[counter,'close0_x':('close'+str(steps))+'_x'].drop(['avg_vol1_x'])])
#plt.plot(test[0],test[1])
#(((test_prices.loc[counter,'close']/test_prices.loc[min(test[0]):max(test[0]),'close'])*100)-100).plot()
#plt.show()


#Split into input and output variables
X = prices1.iloc[:,np.r_[0,5:len(prices1.columns)]].set_index(pd.DatetimeIndex(prices1['date'])).drop('date',1)
Y = prices1.iloc[:,[0,4]].set_index(pd.DatetimeIndex(prices1['date'])).drop('date',1)
train_date = '06-01-2018'

#Split into train and test sets
X_train = X.loc[X.index < train_date, :]
X_test = X.loc[X.index >= train_date, :]
y_train = Y.loc[Y.index < train_date, :]
y_test = Y.loc[Y.index >= train_date, :]



# fit model on training data
model = XGBClassifier(n_jobs = 11, max_depth=5, min_child_weight=60, subsample=0.5, gamma= 10)
model.fit(X_train, y_train)

#save model
#joblib.dump(model, 'xgb_model_btc_testing.pkl')

y_pred = model.predict_proba(X_test)

xgb.plot_importance(model)
plt.show()

#Predict on test set and calculate, accuracy and theoretical profit
#Note accuracy found to be ~ 57%
predictions = y_test
predictions['prob'] = y_pred[:,1]
predictions['pred'] = [round(value) for value in y_pred[:,1]]
predictions = pd.merge(prices1.iloc[:,np.r_[1]],predictions,'right', left_index=True, right_index=True)
predictions['close1'] = predictions['close'].shift(periods=-1)
predictions['change'] = predictions['close1']/predictions['close']
predictions['position'] =  np.where(predictions['prob'] >= .63,1,
                                np.where(predictions['prob'] < .39 ,-1,0))

predictions['position'] = predictions['position'].replace(0,np.nan).ffill().replace(np.nan,0)
predictions['trade'] = np.where(predictions['position'] != predictions['position'].shift(1),1,0)
#predictions['gain'] = np.where(predictions['position']==1,(predictions['change']),(1/(((predictions['change']-1))+1)))
predictions['gain'] = np.where(predictions['position']==1,(predictions['change']),1)
predictions['gain2'] = np.where(predictions['trade'] ==1,predictions['gain']-0.001 ,predictions['gain'] )
predictions['cum_gain'] = np.cumprod(predictions['gain2'])

predictions['position'].mean()

accuracy = skll.accuracy_score(predictions.loc[:,'future_b'], predictions.loc[:,'pred'])
kappa = skll.cohen_kappa_score(predictions.loc[:,'future_b'], predictions.loc[:,'pred'])
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(kappa)

