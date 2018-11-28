import numpy as np
import pandas as pd
import datetime as dt
from linearmodels import PooledOLS 

# load the data
path = '/Users/wangergou/Downloads/'

df = pd.read_pickle(path + 'MM_stock_data_FINAL.pkl')

# store stock tickers 
adjClose = df['Adj Close']
tickers = adjClose.columns.values


estimates = dict()
for ticker in tickers:
    estimates[ticker] = dict()

''' Roll (1984) estimate of the effective spread '''

for ticker in tickers:

    logReturn = np.log(adjClose[ticker] / adjClose[ticker].shift(1)).reset_index()
    logReturn['month'] = logReturn['Date'].apply(lambda x: x.month)
    
    returnCov = logReturn.groupby('month')[ticker].apply(lambda x: x.cov(x.shift(1)))
    
    # For cases where the covariance is positive, the Roll estimate must be set to 0.
    rollEstimate = returnCov.apply(lambda x: 2 * np.sqrt(-x) if x < 0 else 0)
    
    estimates[ticker]['rollEstimate'] = rollEstimate

''' The Abdi and Ranaldo (2017) estimate of the effective '''

Low = df['Low']
High = df['High']

for ticker in tickers:

    dfTemp = np.log(adjClose[ticker]).reset_index()
    dfTemp.columns = ['Date', 'c']
    
    dfTemp['eta'] = np.log((High[ticker] + Low[ticker]) / 2).values
    dfTemp['month'] = dfTemp['Date'].apply(lambda x: x.month)
    
    AbdiCov = dfTemp.groupby('month').apply(lambda x: (x.c - x.eta).cov(x.c - x.eta.shift(1)))

    # For cases where the covariance is negative, the Abdi estimate must be set to 0.    
    abdiEstimate = AbdiCov.apply(lambda x: 2 * np.sqrt(x) if x > 0 else 0)

    estimates[ticker]['abdiEstimate'] = abdiEstimate


''' The Amihud (2002) illiquidity ratio ''' 

Volume = df['Volume']

for ticker in tickers:

    absLogReturn = abs(np.log(adjClose[ticker] / adjClose[ticker].shift(1)))
    
    # Beware of the way to calculate the dollar volume
    dollarVol = Volume[ticker] * adjClose[ticker]
    
    amihudRatio = absLogReturn / dollarVol
    
    amihudRatio = amihudRatio.reset_index()
    amihudRatio['month'] = amihudRatio['Date'].apply(lambda x: x.month)
    
    amihudRatio = amihudRatio.groupby('month')[ticker].mean()
    
    estimates[ticker]['amihudRatio'] = amihudRatio

''' The average daily market cap ''' 

MarketCap = df['MarketCap']

for ticker in tickers:

    avgMarketCap = MarketCap[ticker].reset_index()
    
    avgMarketCap['month'] = avgMarketCap['Date'].apply(lambda x: x.month)
    
    avgMarketCap = avgMarketCap.groupby('month')[ticker].mean()
    
    estimates[ticker]['avgMarketCap'] = avgMarketCap


''' The average daily trading volume (in number of shares) '''

Volume = df['Volume']

for ticker in tickers:

    avgVolume = Volume[ticker].reset_index()
    
    avgVolume['month'] = avgVolume['Date'].apply(lambda x: x.month)
    
    avgVolume = avgVolume.groupby('month')[ticker].mean()
    
    estimates[ticker]['avgVolume'] = avgVolume

''' The daily volatility as the standard deviation of daily stock returns over the month '''

adjClose = df['Adj Close']

for ticker in tickers:

    logReturn = np.log(adjClose[ticker] / adjClose[ticker].shift(1)).reset_index()
    logReturn['month'] = logReturn['Date'].apply(lambda x: x.month)
    
    dailyVol = logReturn.groupby('month')[ticker].std()
    
    estimates[ticker]['dailyVol'] = dailyVol
    
    
'''  The average daily value of the inverse of the price ''' 
# Not sure I understand this. Is it just the avg of the 1 / price ?

adjClose = df['Adj Close']

for ticker in tickers:

    invPrice = (adjClose[ticker] ** (-1)).reset_index()
    
    invPrice['month'] = invPrice['Date'].apply(lambda x: x.month)
    
    invPriceAvg = invPrice.groupby('month')[ticker].mean()
    
    estimates[ticker]['invPriceAvg'] = invPriceAvg


'''  compute the correlation matrix between all the variables ''' 

# We end up with 30 corr matrices? 

# for ticker in tickers:
#     print(pd.DataFrame(estimates[ticker]).corr())
    
# Or use panel data? 

panel = pd.DataFrame()

for ticker in tickers:
    
    tickerData = pd.DataFrame(estimates[ticker])
    tickerData['ticker'] = ticker
    
    panel = panel.append(tickerData)
    
panel = panel.reset_index()    
    
corrVars = [x for x in panel.columns if x not in ['month', 'ticker']]
panel[corrVars].corr()
    
''' run a pooled regression of each of the transaction cost estimates against the three explanatory variables  '''

# There seems to be four explanatory variables? 

regResult = dict()

dependent = ['rollEstimate', 'abdiEstimate', 'amihudRatio']
exog = ['avgVolume', 'avgMarketCap', 'dailyVol', 'invPriceAvg']

panel = panel.set_index(['ticker', 'month'])

for y in dependent:
    mod = PooledOLS(panel[y], panel[exog])
    res = mod.fit()
    regResult[y] = res
    

### 



    
    
    
    
    
    
    
    
    
    
    
