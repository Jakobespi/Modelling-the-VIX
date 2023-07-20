#%%
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm  
import matplotlib.dates as mdates
import matplotlib
from scipy import stats
import pandas_datareader as pdr
import yfinance as yf 
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import warnings
import statsmodels.api as sm
from datetime import datetime
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import register_matplotlib_converters
from pandas.plotting import autocorrelation_plot
from tabulate import tabulate
from scipy.stats import skew, kurtosis
from fredapi import Fred
import json
import requests
import matplotlib.ticker as ticker
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import *
from linearmodels.iv import IV2SLS
from statsmodels.regression.linear_model import RegressionResults
from dateutil.relativedelta import relativedelta
import pytz
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#%%
# Calculating indexes

# assume thart MVP_returns and MVP_HAR_returns exist here - I had the variables stored in jupyter variables :)
MVP_returns_index = (1 + MVP_returns).cumprod() *100  
MVP_HAR_index = (1 + MVP_HAR_returns).cumprod() *100


"Plot the cumulative returns for HAR and MVP"

# Define the LaTeX font for text
fontprops = fm.FontProperties(family='serif', size=9)
# Figure 
fig, ax = plt.subplots(figsize=(6, 4), facecolor='lightgrey')
fig.set_alpha(0.5)
fig.set_edgecolor('b')
ax.plot(MVP_HAR_index.index, MVP_HAR_index, label='HAR', color='black', linewidth=1.3, linestyle='--')
ax.plot(MVP_returns_index.index, MVP_returns_index, label='MVP', color='dimgrey', linewidth=1.3, linestyle='--')

# Altering the y axis
ax.set_ylabel('Index', fontproperties=fontprops, fontsize=9)
ax.tick_params(axis='y', labelsize=9)

# Altering the x axis
years = mdates.YearLocator()
months = mdates.MonthLocator((1,4,7,10))
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(months)
ax.set_xlabel('',fontsize=9)
ax.tick_params(axis='x', labelsize=9)

# Background and legend
ax.legend(fontsize=7, prop=fontprops, edgecolor='black', fancybox=False, shadow=False)
ax.set_facecolor('white')
ax.set_alpha(0.2)
fig.suptitle('Figure 4: MVP vs HAR Cumulative Returns', fontproperties=fontprops, fontsize=12, y=0.95)
plt.savefig('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\Grafer\\Fig4_MVP_vs_HAR.png')
plt.show()



#%% #  Plot the cumulative returns for all Models
fig, axs = plt.subplots(2, 2, figsize=(10,7))

axs[0, 0].plot(MVP_HAR_index.index, MVP_HAR_index, label='HAR returns', color='black', linewidth=1, linestyle='--')
axs[0, 0].plot(MVP_returns_index.index, MVP_returns_index, label='MVP returns', color='grey', linewidth=1, linestyle='--')
axs[0, 0].set_ylabel('Index', fontsize=8)
axs[0, 0].set_title('A: MVP vs HAR', fontsize=8)
axs[0, 0].legend(fontsize=6)
axs[0, 0].tick_params(axis='y', labelsize=8)

axs[0, 0].legend(fontsize=6)
axs[0, 1].plot(MVP_HAR_NI_index.index, MVP_HAR_NI_index, label='HAR-NI returns', color='navy', linewidth=1, linestyle='--')
axs[0, 1].plot(MVP_returns_index.index, MVP_returns_index, label='MVP returns', color='grey', linewidth=1, linestyle='--')
axs[0, 1].set_ylabel('Index', fontsize=8)
axs[0, 1].set_title('B: MVP vs HAR-NI', fontsize=8)
axs[0, 1].legend(fontsize=6)
axs[0, 1].tick_params(axis='y', labelsize=8)

axs[1, 0].plot(MVP_HAR_NIL_index.index, MVP_HAR_NIL_index, label='HAR-NIL returns', color='maroon', linewidth=1, linestyle='--')
axs[1, 0].plot(MVP_returns_index.index, MVP_returns_index, label='MVP returns', color='grey', linewidth=1, linestyle='--')
axs[1, 0].set_ylabel('Index', fontsize=8)
axs[1, 0].set_title('C: MVP vs HAR-NIL', fontsize=8)
axs[1, 0].legend(fontsize=6)
axs[1, 0].tick_params(axis='y', labelsize=8)

axs[1, 1].plot(MVP_HAR_NIL_index.index, MVP_HAR_NIL_index, label='HAR-NIL returns', color='purple', linewidth=1, linestyle='--')
axs[1, 1].plot(MVP_returns_index.index, MVP_returns_index, label='MVP returns', color='grey', linewidth=1, linestyle='--')
axs[1, 1].set_ylabel('Index', fontsize=8)
axs[1, 1].set_title('D: MVP vs HAR-N', fontsize=8)
axs[1, 1].legend(fontsize=6)
axs[1, 1].tick_params(axis='y', labelsize=8)

fig.suptitle('Figure 4: MVP vs Model Cumulative Returns')
plt.savefig('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\Grafer\\Fig4mult_MVP_vs_Models.png')


#%%

# data
start = datetime(2018,1,1)
end = datetime(2023,2,1)

sp500_tick = yf.Ticker('^GSPC')
sp500_hist = sp500_tick.history(start=start, end=end)
sp500 = pd.DataFrame(sp500_hist.loc[:,'Close'])

VIX_tick = yf.Ticker('^VIX')
VIX_hist = VIX_tick.history(start=start, end=end)
VIX = pd.DataFrame(VIX_hist.loc[:,'Close'])

# Bloomberg Inflation numbers - actual and surveyed economist average numbers
Inflation = pd.read_excel('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\DataBehandling\\Expected Inflation.xlsx', sheet_name='Ark1')
Inflation.index = pd.to_datetime(Inflation['Date'])
Inflation.sort_index(inplace=True) # formatting the index
Inflation.rename(columns = {'Actual': 'Inflation'}, inplace=True)
Inflation['InflationDiff'] = Inflation['Inflation'] - Inflation['Surv(A)']  # Creating News Impact series
Inflation['InflationDummy'] = Inflation['InflationDiff'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1) # News Impact Dummy
Inflation = Inflation.loc[: '2023-02-14'] # filtering data wrt. specfied range
Inflation.to_excel('InflationDummy.xlsx')

# Bloomberg Funds Rate numbers -  actual and surveyed economist average numbers 
FundsRate = pd.read_excel('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\DataBehandling\\Expected Funds Rate.xlsx', sheet_name='Ark1')
FundsRate.index = pd.to_datetime(FundsRate['Date'])
FundsRate.sort_index(inplace=True) # formatting the index
FundsRate['FundsRateDiff'] = FundsRate['FundsRate'] - FundsRate['Surv(A)']
FundsRate['FundsRateDummy'] = FundsRate['FundsRateDiff'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1) # News Impact Dummy
FundsRate = FundsRate.loc[: '2023-02-01'] # filtering data wrt. specfied range
FundsRate.to_excel('FundsRateDummy.xlsx')

# Bloomberg Unemployment Rate numbers -  actual and surveyed economist average numbers 
Unemployment = pd.read_excel('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\DataBehandling\\Expected Unemployment.xlsx', sheet_name='Ark1')
Unemployment.index = pd.to_datetime(Unemployment['Date'])
Unemployment.sort_index(inplace=True) # formatting the index
Unemployment['UnemploymentDiff'] = Unemployment['Unemployment'] - Unemployment['Surv(A)']
Unemployment['UnemploymentDummy'] = Unemployment['UnemploymentDiff'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1) # News Impact Dummy
Unemployment = Unemployment.loc[: '2023-02-03'] # filtering data wrt. specfied range
Unemployment.to_excel('UnemploymentDummy.xlsx')

#%%
"Figure 1 VIX and macro statistics over time"

# Initializations
fontprops = fm.FontProperties(family='serif')
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.set_alpha(0.5)
fig.set_edgecolor('b')

# Plot VIX
vix_line = axs[0, 0].plot(VIX.index, VIX['Close'], label='VIX', color='black', linewidth=1.3, linestyle='--')
axs[0, 0].set_ylabel('VIX Close', fontproperties=fontprops)
axs[0, 0].set_ylim([10, VIX['Close'].max()])  # Set y-axis limits to start from 0 and go up to the maximum value of VIX
axs[0, 0].set_title('A: VIX & S&P 500', fontfamily=fontprops)
axs[0, 0].tick_params(axis='y')

# Plot S&P
axs2 = axs[0,0].twinx()
sp500_line = axs2.plot(sp500.index, sp500['Close'], label='S&P 500', color='maroon', linewidth=1.3, linestyle='--')
axs2.set_ylim([2000, sp500['Close'].max()+1000])  # Set y-axis limits to start from 0 and go up to the maximum value of S&P 500
axs2.tick_params(axis='y')
axs2.set_ylabel('S&P500 Close', fontproperties=fontprops)

# Shade the area between March 2020 and 2021
start_date = '2020-02-01'
end_date = '2021-04-01'
VIXmax = VIX['Close'].max()
axs[0, 0].fill_between(VIX.index, 0, VIXmax, where=(VIX.index >= start_date) & (VIX.index <= end_date), alpha=0.2,
                       color='grey')

# Altering the x axis
years = mdates.YearLocator()
axs[0, 0].xaxis.set_major_locator(years)
axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[0, 0].set_xlabel('')
axs[0, 0].tick_params(axis='x')

# Background and legend
lines = vix_line + sp500_line
labels = [line.get_label() for line in lines] 
legend = axs[0,0].legend(lines, labels, fontsize=10, prop=fontprops, edgecolor='black', fancybox=False, shadow=False, loc='upper left', bbox_transform=plt.gcf().transFigure)
axs[0,0].set_facecolor('white')
axs[0,0].set_alpha(0.2)

for text in legend.get_texts():
    text.set_fontsize(8)

# Plot Inflation
axs[0, 1].plot(Inflation.index, Inflation['Inflation'], color='black', label='Inflation', linewidth=1.3, linestyle='--')
axs[0, 1].set_ylabel('CPI MoM', fontproperties=fontprops)
axs[0, 1].set_ylim([Inflation['Inflation'].min(), Inflation['Inflation'].max()]) # Set y-axis limits to follow the range of CPI data
axs[0, 1].set_title('B: Inflation Rate', fontproperties=fontprops)
axs[0, 1].tick_params(axis='y', labelsize=7)
Inflationmax = Inflation['Inflation'].max()
Inflationmin = Inflation['Inflation'].min()
axs[0, 1].fill_between(VIX.index, Inflationmin, Inflationmax, where=(VIX.index>=start_date)&(VIX.index<=end_date), alpha=0.2, color='grey')

# Altering the x axis
years = mdates.YearLocator()
axs[0, 1].xaxis.set_major_locator(years)
axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[0, 1].set_xlabel('')
axs[0, 1].tick_params(axis='x', labelsize=7)

# Plot Fed Funds Rate
axs[1, 0].plot(FundsRate.index, FundsRate['FundsRate'], color='black', label='FundsRate', linewidth=1.3, linestyle='--')
axs[1, 0].set_ylabel('Funds rate (upper bound)', fontproperties=fontprops, fontsize=7)
axs[1, 0].set_ylim([-0.005, FundsRate['FundsRate'].max()]) # Set y-axis limits to follow the range of CPI data)
axs[1, 0].set_title('C: Federal Funds Rate', fontproperties=fontprops, size=7)
axs[1, 0].tick_params(axis='y', labelsize=7)
FEDmax = FundsRate['FundsRate'].max()
axs[1, 0].fill_between(VIX.index, -0.05, FEDmax, where=(VIX.index>=start_date)&(VIX.index<=end_date), alpha=0.2, color='grey')

# Altering the x axis
years = mdates.YearLocator()
axs[1, 0].xaxis.set_major_locator(years)
axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[1, 0].set_xlabel('')
axs[1, 0].tick_params(axis='x', labelsize=7)

# Plot Labour Market statistics
axs[1, 1].plot(Unemployment.index, Unemployment['Unemployment'], color='black', label='Unemployment rate', linewidth=1.3, linestyle='--')
axs[1, 1].set_ylabel('Unemployment rate MoM', fontproperties=fontprops, fontsize=7)
axs[1, 1].set_ylim([Unemployment['Unemployment'].min(), Unemployment['Unemployment'].max()]) # Set y-axis limits to follow the range of CPI data
axs[1, 1].set_title('D: Unemployment Rate', fontproperties=fontprops, fontsize=7)
Unempmax = Unemployment['Unemployment'].max()
axs[1, 1].fill_between(VIX.index, 0, Unempmax, where=(VIX.index>=start_date)&(VIX.index<=end_date), alpha=0.2, color='grey')
axs[1, 1].tick_params(axis='y', labelsize=7)

# Altering the x axis
years = mdates.YearLocator()
axs[1, 1].xaxis.set_major_locator(years)
axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[1, 1].set_xlabel('')
axs[1, 1].tick_params(axis='x', labelsize=6)

fig.set_facecolor('lightgrey')
fig.set_alpha(0.2)
for text in fig.findobj(matplotlib.text.Text):
    text.set_fontproperties(fontprops)
    text.set_fontsize(12)
fig.subplots_adjust(hspace=0.25, wspace=0.5)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95)
fig.text(0.5, 0.02, 'Source: Bloomberg', fontproperties=fontprops, fontsize=12, ha='center')
#fig.suptitle('Figure 1: VIX, S&P 500, and US Macro Statistics Over Time', fontproperties=fontprops, fontsize=16, y=0.96)
plt.savefig('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\Grafer\\Fig1.png')
plt.show()

#%%
"Plot ACF and PACF"

fig, axs = plt.subplots(2, 1, figsize=(12,8))

# Plot ACF
acf = plot_acf(VIX['Close'], lags=40, ax=axs[0])
axs[0].set_title('Autocorrelation Function (ACF)', y=1.03)
axs[0].tick_params(axis='x', labelsize=10)
axs[0].tick_params(axis='y', labelsize=10)
axs[0].set_xlabel('Lag', fontsize= 10,fontproperties=fontprops)
axs[0].set_ylabel('ACF', fontsize= 10,fontproperties=fontprops)
axs[0].set_ylim([-0.4,1.2])

# Plot PACF
plot_pacf(VIX['Close'], lags=40, ax=axs[1])
axs[1].set_title('Partial Autocorrelation Function (PACF)', y=1.03)
axs[1].tick_params(axis='x', labelsize=10)
axs[1].tick_params(axis='y', labelsize=10)
axs[1].set_xlabel('Lag', fontsize= 10,fontproperties=fontprops)
axs[1].set_ylabel('PACF', fontsize= 10,fontproperties=fontprops)
axs[1].set_ylim([-0.2,1.2])



plt.subplots_adjust(hspace=0.48)
fig.subplots_adjust(left=0.098, right=0.98, bottom=0.2, top=0.93)
fig.set_facecolor('lightgrey')
fig.set_alpha(0.2)
for text in fig.findobj(matplotlib.text.Text):
    text.set_fontproperties(fontprops)
    text.set_fontsize(16)

fig.text(0.5, 0.02, 'The autocorrelation and partial autocorrelation functions are derived from the sample of the VIX index  \n' 
         'from January 1, 2018 to February 2, 2023. The blue band refers to the 95% confidence interval under\n'
         'the null of zero autocorrelation.', 
         ha='center', fontproperties=fontprops, fontsize=15,)
plt.savefig('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\Grafer\\ACF_PACF.png')
plt.show()