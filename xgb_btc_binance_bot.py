import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import os
from binance.client import Client
import time
import threading
import datetime
import joblib
#os.chdir('C:/Users/HGuessous/Documents/Hakim Stuff/crypto')

#Set API keys
api_key = 'insert'
api_secret = 'insert'

client = Client(api_key, api_secret)
symbol = 'BTCUSDT'

#load XGB model created from Model_btc.py
model = joblib.load('xgb_model_btc.pkl')

#Get recent prices from Binance client
def get_prices():
    btc = client.get_historical_klines(symbol=symbol, interval='5m', start_str='3 days ago UTC')
    btc = pd.DataFrame(btc)
    btc.columns = ['Open time', 'Open', 'High', 'Low', ' Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'taker buy base asset volume', 'Taker buy quote asset volume', 'ignore']
    btc['Open time'] = pd.to_datetime(btc['Open time'], unit='ms')
    btc['Close time'] = pd.to_datetime(btc['Close time'], unit='ms')
    btc = btc[['Open time', 'Open', 'High', 'Low', ' Close', 'Volume']]


    btc.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    btc = btc.set_index('date')

    btc['open'] = btc['open'].astype(float)
    btc['close'] = btc['close'].astype(float)
    btc['high'] = btc['high'].astype(float)
    btc['low'] = btc['low'].astype(float)
    btc['volume'] = btc['volume'].astype(float)

    btc = (btc.resample('10T')
         .agg({'open': 'first', 'close': 'last',
               'high': np.max, 'low': np.min,
               'volume': np.sum}))
    btc = btc.dropna()
    return(btc)



#Convert prices data into min/max structure
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


#Convert min/max structure into model format for predictions
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
    pricesdata.iloc[:,4:len(pricesdata.columns)] = pricesdata.iloc[:,4:len(pricesdata.columns)].fillna(method='ffill').shift(periods=order)
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

#Create prediction set
def get_data(prices, smoothing, window_range, order, steps):
    minmax = get_max_min(prices, smoothing, window_range, order)
    prices = prices.reset_index()
    prices1 = get_trend(prices, minmax, order, steps)

    return prices1


#Runs trading algorithm which takes prediction outputs to place buy, sell orders or holds current position
def run_forecast():
    print(datetime.datetime.now(),'Getting data from binance')

    order = 5
    window_range = order
    smoothing = 3
    steps = 10

    prices = get_prices()
    prices = prices.drop(['open', 'high', 'low'], axis=1)
    prices['avg_vol'] = prices['volume'].rolling(window=int(order / 2)).mean()

    prices1 = get_data(prices, smoothing, window_range, order, steps)
    prices1 = prices1.set_index(pd.DatetimeIndex(prices1['date']))
    X = prices1.iloc[:, np.r_[0, 3:len(prices1.columns)]].set_index(pd.DatetimeIndex(prices1['date'])).drop('date', 1)
    X = X.iloc[0: len(X) - 1,:]
    X = X.tail(1)

    print(datetime.datetime.now(), 'Predicting state')
    y_pred = model.predict_proba(X)[:, 1]

    predictions = pd.DataFrame({'pred':y_pred,'price':prices.iloc[prices.index == prices.index.max()]['close']})
    predictions['pred'] = np.where(predictions['pred'] > .62, 1,np.where(predictions['pred'] < .41,-1,0))


    #Get Current Position
    current_price = float(prices.iloc[prices.index == prices.index.max()]['close'])

    for i in range(1,4):
        account = client.get_account()
        account = pd.DataFrame(account['balances'])
        account = account.loc[account['asset'].isin(['BTC', 'USDT']),:]
        position = str(np.where(float(account[account['asset']=='BTC']['free'])*(current_price/10) > float(account[account['asset']=='USDT']['free']), 'BTC',
                            np.where(float(account[account['asset']=='BTC']['free'])*(current_price) < float(account[account['asset']=='USDT']['free'])/10,'USDT','Mixed')))

        current_pred = predictions.loc[predictions.index.max(),'pred']
        action = np.where((position == 'BTC' or position == 'Mixed') and (current_pred == -1), 'Sell',
                          np.where((position == 'USDT' or position == 'Mixed') and (current_pred == 1), 'Buy', 'Hold'))
        if i == 1:
            print(datetime.datetime.now(),current_price,'Predition:',round(float(y_pred),3) ,'-', current_pred,'   Position:', position,'  Action', action)

        if action != 'Hold':

            if action == 'Sell':
                BTC = client.get_orderbook_ticker(symbol='BTCUSDT')
                bid = BTC["bidPrice"]
                ask = BTC["askPrice"]
                price = str(round(((float(bid)) + (float(ask)*4))/5,2))
                price = price.ljust(13, '0')

                quantity = round(float(account[account['asset']=='BTC']['free'])*.999, 6)
                print(datetime.datetime.now(),'Attempting to sell', quantity,'BTC at',price)
                client.order_limit_sell(timeInForce='GTC', symbol='BTCUSDT', quantity=quantity, price=price)


            if action == 'Buy':
                BTC = client.get_orderbook_ticker(symbol='BTCUSDT')
                bid = BTC["bidPrice"]
                ask = BTC["askPrice"]
                price = str(round(((float(bid)*4) + float(ask))/5,2))
                price = price.ljust(13, '0')

                quantity = round(float(account[account['asset'] == 'USDT']['free']) * .999 /round(((float(bid)*4) + float(ask))/5,2) , 6)
                print(datetime.datetime.now(),'Attempting to buy', quantity, 'BTC at', price)
                client.order_limit_buy(timeInForce='GTC', symbol='BTCUSDT', quantity=quantity, price=price)


            print(datetime.datetime.now(),'waiting 1 minutes')
            time.sleep(60)

            open_orders = client.get_open_orders(symbol='BTCUSDT')
            if open_orders != []:
                open_orders = open_orders[0]['orderId']
                print(datetime.datetime.now(),'order',open_orders,'still open after 1 minutes, will cancel order and try again.')
                try:
                    client.cancel_order(symbol='BTCUSDT', orderId=open_orders)
                except:
                    print("could not cancel order")
                print(datetime.datetime.now(),action,'attempt',i,'failed')

            if open_orders == []:
                trades = client.get_my_trades(symbol='BTCUSDT')
                trades = trades[len(trades)-1]
                print(datetime.datetime.now(),action, 'attempt', i, 'successful',trades['qty'],'BTC',action,'at',trades['price'])

        if action == 'Hold' and i ==1:
            print(datetime.datetime.now(),action,'- No action necessary')



#Times model to run every 10 minutes
t = None
def startBot():
    global t
    run_forecast()
    print(datetime.datetime.now(),'Sleeping -', 10 - datetime.datetime.now().minute % 10, 'minutes')
    time.sleep((10 - datetime.datetime.now().minute % 10)*60)
    t = threading.Timer(5, startBot)
    t.start()

startBot()

#t.cancel()



