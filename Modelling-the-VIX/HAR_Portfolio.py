#%% # Imports
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy import stats
import pandas_datareader as pdr
import yfinance as yf 
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import warnings
import statsmodels.api as sm
import datetime
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
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

def calculate_mvp_weights(cov_matrix):
    ones = np.ones(len(cov_matrix))
    weights = np.linalg.inv(cov_matrix).dot(ones) / ones.T.dot(np.linalg.inv(cov_matrix)).dot(ones)
    return weights

def calculate_mvp(VIXY_SP_returns, current_month):
    cov_matrix = VIXY_SP_returns.cov()
    mvp_weights = calculate_mvp_weights(cov_matrix)

    VIXY_SP_returns_month = VIXY_SP_returns.loc[current_month:current_month + relativedelta(months=1) - relativedelta(days=1)]
    cumulative_returns_month = (1 + VIXY_SP_returns_month).cumprod() - 1

    mvp_returns = np.dot(cumulative_returns_month.iloc[-1], mvp_weights)

    return mvp_weights, mvp_returns

def calculate_altered_mvp(mvp_weights_list, util_diff, VIXY_SP_returns, current_month, altered_mvp_df):
    new_weights = np.zeros(2)
    if util_diff > 0.01:
        new_weights[0] = 1.1 # VIX weight altered
        new_weights[1] = -0.1  # S&P weight unchanged
    else:
        new_weights = mvp_weights_list[-1]  # same as MVP weights
    #new_weights = new_weights / new_weights.sum()

    # Calculate altered MVP returns
    VIXY_SP_returns_month = VIXY_SP_returns.loc[current_month:current_month + relativedelta(months=1) - relativedelta(days=1)]
    cumulative_returns_month = ((1 + VIXY_SP_returns_month).prod()) - 1
    altered_mvp_returns = np.dot(cumulative_returns_month, new_weights)

    # Update altered MVP DataFrame
    altered_mvp_df.loc[current_month, 'VIXY_return'] = cumulative_returns_month[0]
    altered_mvp_df.loc[current_month, 'SP_return'] = cumulative_returns_month[1]
    altered_mvp_df.loc[current_month, 'Altered_Weight_VIXY'] = new_weights[0]
    altered_mvp_df.loc[current_month, 'Altered_Weight_SP'] = new_weights[1]
    altered_mvp_df.loc[current_month, 'Util>0?'] = util_diff > 0.01
    
    return altered_mvp_df, altered_mvp_returns, new_weights

 # Loop
"""
For each month in the data sample i am:
1. Updating the VIX, S&P and VIXY data
2. Calculating the MVP
3. Calculating the probability of an increase in VIX next month: pi(m)
4. Calculating expected long and short utilities of holding the VIX
4. Deriving the MVP containing VIXY and S&P 500
5. Deriving whether the expected utility of being long is greater than expected utility of being short
6. Increasing VIXY weight if long utility is greater, and vice verca.
"""
# Initializations
data_start = datetime(2015,1,1)
data_end = datetime(2023,2,1)
HAR_pi = pd.DataFrame(columns=['month', 'pi'])
MVP_weights_df = pd.DataFrame(columns=['VIXY', 'S&P'])
returns_summary = pd.DataFrame(columns=['Month', 'VIX_Positive_days', 'VIX_Negative_days',
                                        'Long_LogR_VIX_positive', 'Long_LogR_VIX_negative',
                                        'SP_Positive_days', 'SP_Negative_days',
                                        'Long_LogR_SP_positive', 'Long_LogR_SP_negative',
                                        'Short_LogR_VIX_positive', 'Short_LogR_VIX_negative',
                                        'Short_LogR_SP_positive', 'Short_LogR_SP_negative'])
start_HAR = False
start_MVP_model = False
mvp_returns_dict = {}
altered_mvp_returns_dict = {}
mvp_cumulative_returns_dict = {}
altered_mvp_cumulative_returns_dict = {}
mvp_weights_list = []
altered_mvp_weights_list = []
altered_mvp_returns_list = []
mvp_cumulative_returns = 100
altered_mvp_cumulative_returns = 100
VIXY_cumulative_returns = 100
SP_cumulative_returns = 100
total_long_months = 0
total_short_months = 0
delta = relativedelta(data_end, data_start)
num_months = delta.years * 12 + delta.months
previous_util_diff = None
altered_mvp_df = pd.DataFrame(columns=['VIXY_return', 'SP_return', 'Altered_Weight_VIXY', 'Altered_Weight_SP', 'Util>0?'])

for i in tqdm(range(num_months), desc="Processing"):
    current_month = data_start + relativedelta(months=i)
    current_month = pytz.timezone('America/New_York').localize(current_month)

    "Updating VIX, S&P and VIXY data"
    VIX = yf.Ticker('^VIX')
    VIX_data = VIX.history(start=current_month, end=current_month+relativedelta(months=1)-relativedelta(days=1))
    VIX_data = pd.DataFrame(VIX_data.loc[:, 'Close'])
    VIX_data.rename(columns={'Close': 'VIX_d'}, inplace=True)

    SP_tick = yf.Ticker('SPY')
    SP_data = SP_tick.history(start=current_month, end=current_month+relativedelta(months=1)-relativedelta(days=1))
    SP_data = pd.DataFrame(SP_data.loc[:, 'Close'])
    SP_data.rename(columns={'Close':'S&P'}, inplace =True) 
    
    VIXY_tick = yf.Ticker('VIXY')
    VIXY_data = VIXY_tick.history(start=current_month, end=current_month+relativedelta(months=1)-relativedelta(days=1))
    VIXY_data = pd.DataFrame(VIXY_data.loc[:, 'Close'])
    VIXY_data.rename(columns={'Close':'VIXY'}, inplace =True) 

    "Extend the VIX, S&P and VIXY data each month"
    if i == 0:
        all_VIX_data = VIX_data
        all_SP_data = SP_data
        all_VIXY_data = VIXY_data
    else:
        all_VIX_data =  pd.concat([all_VIX_data, VIX_data])
        all_SP_data =  pd.concat([all_SP_data, SP_data])
        all_VIXY_data = pd.concat([all_VIXY_data, VIXY_data])
    
    all_VIX_data['VIX_return']= all_VIX_data['VIX_d'].pct_change()
    all_SP_data['S&P_return']= all_SP_data['S&P'].pct_change()
    all_VIXY_data['VIXY_return']= all_VIXY_data['VIXY'].pct_change()
    
    "Monthly rebalanced MVP consisting of S&P and VIXY."
    if current_month >= pytz.timezone('America/New_York').localize(datetime(2020, 1, 1)):
        VIXY_SP_returns = pd.concat([all_VIXY_data['VIXY_return'], all_SP_data['S&P_return']], axis=1)
        VIXY_SP_returns.dropna(inplace=True)
        
        # monthly returns and weights
        mvp_weights, mvp_returns = calculate_mvp(VIXY_SP_returns, current_month)
        mvp_weights_list.append(mvp_weights)
        mvp_returns_dict[current_month] = mvp_returns
        
        # cumulative returns
        if current_month == pytz.timezone('America/New_York').localize(datetime(2020, 1, 1)):
            mvp_cumulative_returns_dict[current_month] = 100
        else: 
            mvp_cumulative_returns *= (1 + mvp_returns)
            mvp_cumulative_returns_dict[current_month] = mvp_cumulative_returns

    "Update HAR model each month after training period  :2020, 1, 1"
    # Updating HAR components each month
    all_VIX_data['VIX_1'] = all_VIX_data['VIX_d'].shift(1)
    all_VIX_data['VIX_w'] = all_VIX_data['VIX_1'].rolling(5).mean()
    all_VIX_data['VIX_m'] = all_VIX_data['VIX_1'].rolling(22).mean()
    
    if current_month >= pytz.timezone('America/New_York').localize(datetime(2020,1,1)):
        HAR_IS = all_VIX_data.copy()
        HAR_IS.drop(columns=['VIX_return'])
        HAR_IS.dropna(inplace=True)
        HAR_IS['Date'] = HAR_IS.index
        HAR_IS.index = pd.to_datetime(HAR_IS.index)
        HAR_IS_results = sm.OLS.from_formula(formula='VIX_d ~ VIX_1+VIX_w+VIX_m',
                                          data=HAR_IS).fit(cov_type='HC0')
        HAR_IS_params = HAR_IS_results.params
        HAR_IS_resid = HAR_IS_results.resid
        HAR_IS_ecdf = sm.distributions.ECDF(HAR_IS_resid) # updated each month
        HAR_IS['predicted'] = HAR_IS_results.predict()
        HAR_IS['delta'] = HAR_IS['VIX_1'] - HAR_IS['predicted']
        HAR_IS['pi'] = 1- HAR_IS_ecdf(HAR_IS['delta']) # chance of increased VIX next day

        # Calculate pi and 22 day mean delta each month
        delta_22_mean = HAR_IS.loc[(HAR_IS.index[-1]>= current_month) & (HAR_IS.index < current_month- relativedelta(months=1)-relativedelta(days=1)), 'delta'].mean()
        pi = 1- HAR_IS_ecdf(delta_22_mean)
        # Add pi and delta_22_mean to HAR_pi dataframe
        HAR_pi.loc[current_month, 'month'] = current_month
        HAR_pi.loc[current_month, 'pi'] = pi

    "Calculate expected utilities each month"
    month_start_date = current_month - relativedelta(months=1)
    VIX_month_long_data = all_VIX_data.loc[month_start_date:current_month]
    SP_month_long_data = all_SP_data.loc[month_start_date:current_month]

    # Number of positive and negative return days and average log returns for long position
    VIX_positive_returns = VIX_month_long_data['VIX_return'][VIX_month_long_data['VIX_return'] > 0]
    VIX_negative_returns = VIX_month_long_data['VIX_return'][VIX_month_long_data['VIX_return'] < 0]
    VIX_positive_log_returns = np.mean(np.log(1 + VIX_positive_returns))
    VIX_negative_log_returns = np.mean(np.log(1 + VIX_negative_returns))
    
    SP_positive_returns = SP_month_long_data['S&P_return'][SP_month_long_data['S&P_return'] > 0]
    SP_negative_returns = SP_month_long_data['S&P_return'][SP_month_long_data['S&P_return'] < 0]
    SP_positive_log_returns = np.mean(np.log(1 + SP_positive_returns))
    SP_negative_log_returns = np.mean(np.log(1 + SP_negative_returns))

    # Average log returns when shorting VIX/SPY on days they fell/rose respectively
    Short_VIX_positive_log_returns = np.mean(np.log(1 - VIX_positive_returns))
    Short_VIX_negative_log_returns = np.mean(np.log(1 - VIX_negative_returns))
    Short_SP_positive_log_returns = np.mean(np.log(1 - SP_positive_returns))
    Short_SP_negative_log_returns = np.mean(np.log(1 - SP_negative_returns))

    # Append the results to returns_summary dataframe
    returns_summary = returns_summary.append({'Month': current_month, 
                                           'VIX_Positive_days': len(VIX_positive_returns), 
                                           'VIX_Negative_days': len(VIX_negative_returns), 
                                           'Long_LogR_VIX_positive': VIX_positive_log_returns, 
                                           'Long_LogR_VIX_negative': VIX_negative_log_returns, 
                                           'SP_Positive_days': len(SP_positive_returns), 
                                           'SP_Negative_days': len(SP_negative_returns), 
                                           'Long_LogR_SP_positive': SP_positive_log_returns, 
                                           'Long_LogR_SP_negative': SP_negative_log_returns, 
                                           'Short_LogR_VIX_positive': Short_VIX_positive_log_returns, 
                                           'Short_LogR_VIX_negative': Short_VIX_negative_log_returns,  
                                           'Short_LogR_SP_positive': Short_SP_positive_log_returns, 
                                           'Short_LogR_SP_negative': Short_SP_negative_log_returns}, 
                                          ignore_index=True)
    returns_summary['Month'] = pd.to_datetime(returns_summary['Month'])
    returns_summary = returns_summary.loc[returns_summary['Month'] >= '2020-01-01']

    # Calculate expected VIX utilities
    Utility = pd.concat([returns_summary.set_index('Month'), HAR_pi], axis=1, join='outer')
    Utility['U_long'] = Utility['pi'] * Utility['Long_LogR_VIX_positive'] + (1-Utility['pi'])*Utility['Long_LogR_VIX_negative']
    Utility['U_short'] = Utility['pi'] * Utility['Short_LogR_VIX_positive'] + (1-Utility['pi'])*Utility['Short_LogR_VIX_negative']
    Utility['U_diff'] = Utility['U_long'] - Utility['U_short']

    "Monthly rebalanced MVP that is altered based on U_diff "
    
    if current_month >= pytz.timezone('America/New_York').localize(datetime(2020, 1, 1)):
        util_diff = Utility.loc[current_month, 'U_diff']
        altered_mvp_df, altered_mvp_returns, new_weights = calculate_altered_mvp(
                mvp_weights_list, util_diff, VIXY_SP_returns, current_month, altered_mvp_df)
        print(new_weights)


# Ensuring that the returns are calculated correctly
altered_mvp_df['Date'] = altered_mvp_df.index
altered_mvp_df = altered_mvp_df.reset_index(drop=True)
altered_mvp_df.iloc[2:, altered_mvp_df.columns.get_loc('Altered_Weight_VIXY')] = altered_mvp_df.iloc[2:, altered_mvp_df.columns.get_loc('Altered_Weight_VIXY')].shift(1)
altered_mvp_df.iloc[2:, altered_mvp_df.columns.get_loc('Altered_Weight_SP')] = altered_mvp_df.iloc[2:, altered_mvp_df.columns.get_loc('Altered_Weight_SP')].shift(1)
altered_mvp_df.iloc[2, altered_mvp_df.columns.get_loc('Altered_Weight_VIXY')] = mvp_weights_list[2][0]
altered_mvp_df.iloc[2, altered_mvp_df.columns.get_loc('Altered_Weight_SP')] = mvp_weights_list[2][1]
altered_mvp_df['VIXY_return'] = altered_mvp_df.apply(
    lambda row: row['VIXY_return'] * np.sign(row['Altered_Weight_VIXY']), axis=1
)
altered_mvp_df['SP_return'] = altered_mvp_df.apply(
    lambda row: row['SP_return'] * np.sign(row['Altered_Weight_SP']), axis=1
)
# Calculate HAR return
altered_mvp_df['HAR_return'] = (altered_mvp_df['VIXY_return'] * altered_mvp_df['Altered_Weight_VIXY'] +
                               altered_mvp_df['SP_return'] * altered_mvp_df['Altered_Weight_SP'])

# Export DataFrame to Excel
#altered_mvp_df.index.tz_localize(None)
altered_mvp_df['Date'] = altered_mvp_df['Date'].dt.tz_localize(None)
altered_mvp_df.to_excel('HAR-3_df.xlsx', index=True)
#%%
HAR_cum = pd.read_excel('HAR-1_df.xlsx')
HAR_cum.rename(columns={'Unnamed: 8':'cum'}, inplace=True)

HAR2_cum = pd.read_excel('HAR-2_df.xlsx')
HAR2_cum.rename(columns={'Unnamed: 8':'cum'}, inplace=True)
"Illustrating Results" 

# Months where long and short are recommended
long_short_months = pd.DataFrame(Utility['U_diff']>0.01)
long_num = long_short_months.loc[long_short_months['U_diff']==True].sum()
short_num = long_short_months.loc[long_short_months['U_diff']==False].sum()

# Convert dictionaries to pandas Series
MVP_returns_index = pd.Series(mvp_cumulative_returns_dict)
MVP_HAR_returns_index = HAR_cum[['Date', 'cum']]
MVP_HAR2_returns_index = HAR2_cum[['Date', 'cum']]
#MVP_HAR_returns_index.set_index(MVP_HAR_returns_index['Date'], inplace=True)
long_bar = pd.Series(0, index=MVP_HAR2_returns_index.index)
long_short_months.index = MVP_HAR2_returns_index.index
for idx, row in long_short_months.iterrows():
    if row['U_diff']:
        long_bar[idx] = MVP_HAR2_returns_index.loc[idx, 'cum']

"S&P and VIX"
#%%
VIXY = HAR2_cum[['Date', 'VIXY_return']]
#VIXY['VIXY_return'] = VIXY['VIXY_return'].shift(1)
#VIXY['VIXY_return'].fillna(0, inplace=True)
monthly_std = SP['S&P_return'].resample('M').std()*100
# Define the LaTeX font for text
fontprops = fm.FontProperties(family='serif', size=14)
#%%
# Figure 
fig, ax = plt.subplots(figsize=(12, 8), facecolor='lightgrey')
fig.set_alpha(0.5)
fig.set_edgecolor('b')

#left axis
ax2 = ax.twinx()
#ax2.bar(monthly_std.index, monthly_std, color='maroon', width=5, alpha=0.5, label='S&P std')
ax2.bar(VIXY['Date'], VIXY['VIXY_return']*100, color='maroon', width=5, alpha=0.8, label='VIXY return')
ax2.set_ylabel('Returns (%)', fontproperties=fontprops, fontsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14, length=8, which='major')
ax2.tick_params(axis='y', length=6, which='minor')
ax2.set_ylim([-40,HAR2_cum['VIXY_return'].max()*100+5])

# right axis
ax.bar(MVP_HAR2_returns_index['Date'], long_bar,  color='cornflowerblue', width=7, alpha=0.6, label='$\psi>0.02$')
ax.plot(MVP_HAR2_returns_index['Date'], MVP_HAR2_returns_index['cum'], label='HAR-2', color='mediumblue', linewidth=2, linestyle='--')
ax.plot(MVP_HAR_returns_index['Date'], MVP_HAR_returns_index['cum'], label='HAR-1', color='black', linewidth=2, linestyle='--')
ax.plot(MVP_returns_index.index, MVP_returns_index, label='MVP', color='dimgrey', linewidth=2, linestyle='--')

# Altering the y axis
ax.set_ylabel('Index', fontproperties=fontprops, fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='y',labelsize=14, length=8, which='major')
ax.tick_params(axis='y', length=6, which='minor')
ax.yaxis.set_minor_locator(MultipleLocator(20))

# Altering the x axis
years = mdates.YearLocator()
months = mdates.MonthLocator((1,2,3,4,5,6,7,8,9,10,11,12))
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(months)
ax.set_xlabel('')
ax.tick_params(axis='x',labelsize=14, length=8, which='major')
ax.tick_params(axis='x', length=6, which='minor')
for tick in ax.get_xticklabels():
    tick.set_fontname('serif')
for tick in ax.get_yticklabels():
    tick.set_fontname('serif')

# filling recession period
HAR_max = MVP_HAR2_returns_index['cum'].max()
HAR_min = MVP_HAR2_returns_index['cum'].min()
fill_start = datetime(2020,2,1)
fill_end = datetime(2021,4,1)
ax.set_ylim([20,HAR_max+10])
ax.fill_between(MVP_HAR2_returns_index['Date'], HAR_max+10, where=(MVP_HAR2_returns_index['Date']>=fill_start)&(MVP_HAR2_returns_index['Date']<=fill_end), alpha=0.1, color='grey')

# Background and legend
#ax.legend(fontsize=14, edgecolor='black', fancybox=False, shadow=False, loc='center right')
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, fontsize=14, edgecolor='black', fancybox=False, shadow=False, loc='center right')
ax.set_facecolor('white')
ax.set_alpha(0.2)
fig.subplots_adjust(left=0.08, right=0.92, bottom=0.07, top=0.98)
#fig.suptitle('Figure 4: MVP vs HAR Cumulative Returns', fontproperties=fontprops, fontsize=12, y=0.95)
plt.savefig('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\Grafer\\Fig4_MVP_vs_HAR.png')
plt.show()

#%%
MVP_annum_returns = pd.Series(mvp_returns_dict.values(), index=pd.to_datetime(list(mvp_returns_dict.keys())))
MVP_annum_returns_df = MVP_annum_returns.to_frame(name="returns")

# Calculate the annual returns
mvp_annual_returns = MVP_annum_returns_df.groupby(MVP_annum_returns_df.index.year).apply(lambda x: (x + 1).prod() - 1)

# Calculate the 2022 YTD return
current_year = datetime.now().year
ytd_return = MVP_annum_returns_df.loc[str(current_year)].apply(lambda x: (x + 1).prod() - 1)

# Print the results
print("MVP annualized returns:")
print(mvp_annual_returns)

print(f"\n{current_year} YTD return:")
print(ytd_return)


# HAR-1
HAR_cum.set_index('Date', inplace=True)
HAR_annual_returns = (HAR_cum['cum'].resample('Y').last() / HAR_cum['cum'].resample('Y').first()) - 1

# Calculate the 2022 YTD return
#current_year = datetime.now().year
#HAR_ytd_return = HAR_annum_returns_df.loc[str(current_year)].apply(lambda x: (x + 1).prod() - 1)

# Print the results
print("HAR annualized returns:")
print(HAR_annual_returns)

# HAR-2
HAR2_cum.set_index('Date', inplace=True)
HAR2_annual_returns = (HAR2_cum['cum'].resample('Y').last() / HAR2_cum['cum'].resample('Y').first()) - 1

# Calculate the 2022 YTD return
#current_year = datetime.now().year
#HAR_ytd_return = HAR_annum_returns_df.loc[str(current_year)].apply(lambda x: (x + 1).prod() - 1)

# Print the results
print("HAR2 annualized returns:")
print(HAR2_annual_returns)

#print(f"\n{current_year} YTD return:")
#print(HAR_ytd_return)