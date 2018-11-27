import numpy as np
import pandas as pd
import datetime as dt
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

''' FUNCTIONS '''

def getStockData(tickers, start, end):
    
        yf.pdr_override()
        
        df = pdr.get_data_yahoo(tickers, start, end)
        
        return df 

''' END OF FUNCTIONS '''

start = dt.datetime(2017, 1, 1)
end = dt.datetime(2018, 1, 1)

# The definition of small/medium/large caps comes from yahoo
# https://finance.yahoo.com/screener/unsaved/665cba69-7bba-439f-95ce-a643cb8ad730?offset=0&count=100

'''
medium caps:

KYN	Kayne Anderson MLP/Midstream Investment Company
DY	Dycom Industries, Inc.
AY	Atlantica Yield plc
GWLLF	Great Wall Motor Company Limited
OPK	OPKO Health, Inc.
MEDP	Medpace Holdings, Inc.
FIX	Comfort Systems USA, Inc.
BEL	Belmond Ltd.
BZUN	Baozun Inc.
VGR	Vector Group Ltd.
'''

smallCapTickers = ['KYN', 'DY', 'AY', 'GWLLF', 'OPK', 'MEDP', 'FIX', 'BEL', 'BZUN', 'VGR']

'''
medium caps:

M	Macy's, Inc.
OCPNY	Olympus Corporation
BURBY	Burberry Group plc
MAS	Masco Corporation
SUI	Sun Communities, Inc.
AGRPY	Absa Group Limited
YNDX	Yandex N.V.
RZB	Reinsurance Group of America, Incorporated
JEC	Jacobs Engineering Group Inc.
SGEN	Seattle Genetics, Inc.
'''

mediumCapTickers = ['M', 'OCPNY', 'BURBY', 'MAS', 'SUI', 'AGRPY', 'YNDX', 'RZB', 'JEC', 'SGEN']

'''
large caps:

PYPL: PayPal Holdings, Inc.
SFTBY: SoftBank Group Corp.
MS: Morgan Stanley
FDX: FedEx Corporation
TSLA: Tesla, Inc.
NVDA: NVIDIA Corporation
REGN: Regeneron Pharmaceuticals, Inc.
TRV: The Travelers Companies, Inc.
EQIX: Equinix, Inc. (REIT)
NTES: NetEase, Inc.
'''

largeCapTickers = ['PYPL', 'SFTBY', 'MS', 'FDX', 'TSLA', 'NVDA', 'REGN', 'TRV', 'EQIX', 'NTES']

# Load and save stock data 

df = getStockData(smallCapTickers + mediumCapTickers + largeCapTickers, start, end)
df.to_pickle("MM_stock_data.pkl")

df = pd.read_pickle('MM_stock_data.pkl')

# store stock tickers 
tickers = adjClose.columns.values

estimates = dict()
for ticker in tickers:
    estimates[ticker] = dict()

''' Roll (1984) estimate of the effective spread '''

adjClose = df['Adj Close']

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

for ticker in tickers:
    print(pd.DataFrame(estimates[ticker]).corr())











