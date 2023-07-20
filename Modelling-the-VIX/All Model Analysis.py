
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
import arch.unitroot as ur

#%% # Data Collection

"Diverse data indhendt"
# start = datetime(2017,10,27)
start = datetime(2018,1,1)
start_CPI = datetime(2017,1,1)
end = datetime(2023,2,1)

# FRED API
fred = Fred(api_key='b177ac3b610db086e81fea931dcc3cbc')

# FRED Inflation Numbers - Urban CPI, Expected inflation numbers (University of Michigan)
# Inflation = pd.DataFrame({'CPI': fred.get_series('CPIAUCSL', start_CPI, end), 'ExpInflation': fred.get_series('MICH', start, end) })
# Inflation['Inflation'] = Inflation['CPI'].pct_change(periods=12)*100 # Calculate monthly percentage change
# Inflation.dropna(inplace=True)

# Bloomberg Inflation numbers - actual and surveyed economist average numbers
Inflation = pd.read_excel('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\DataBehandling\\Expected Inflation.xlsx', sheet_name='Ark1')
Inflation.index = pd.to_datetime(Inflation['Date'])
Inflation.sort_index(inplace=True) # formatting the index
Inflation.rename(columns = {'Actual': 'Inflation'}, inplace=True)
Inflation['InflationDiff'] = Inflation['Inflation'] - Inflation['Surv(A)']  # Creating News Impact series
Inflation['InflationDummy'] = Inflation['InflationDiff'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1) # News Impact Dummy
Inflation = Inflation.loc[: '2023-02-14'] # filtering data wrt. specfied range
Inflation.to_excel('InflationDummy.xlsx')

# FRED Numbers - FED Funds Rate & FED upper limit target rate
# FundsRate = pd.DataFrame({'FundsRate': fred.get_series('EFFR', start, end), 'ExpFundsRate':fred.get_series('DFEDTARU', start, end) })
# FundsRate.dropna(inplace=True)
# FundsRate = FundsRate[FundsRate.index.is_month_end]

# Bloomberg Funds Rate numbers -  actual and surveyed economist average numbers 
FundsRate = pd.read_excel('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\DataBehandling\\Expected Funds Rate.xlsx', sheet_name='Ark1')
FundsRate.index = pd.to_datetime(FundsRate['Date'])
FundsRate.sort_index(inplace=True) # formatting the index
FundsRate['FundsRateDiff'] = FundsRate['FundsRate'] - FundsRate['Surv(A)']
FundsRate['FundsRateDummy'] = FundsRate['FundsRateDiff'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1) # News Impact Dummy
FundsRate = FundsRate.loc[: '2023-02-01'] # filtering data wrt. specfied range
FundsRate.to_excel('FundsRateDummy.xlsx')

# FRED numbers - US Jobless Claims
# Claims = pd.DataFrame({'Claims': fred.get_series('ICSA', start, end)})

# FRED numbers - US unemployment rate
# Unemployment = pd.DataFrame({'Unemployment': fred.get_series('UNRATE', start, end), 'ExpUnemployment': fred.get_series('UNRATEEEXP1MFRBDAL', start, end)})
# Unemployment['ExpUnemployment'].fillna(method='bfill', inplace=True)
# Unemployment.dropna(inplace=True)

# Bloomberg Unemployment Rate numbers -  actual and surveyed economist average numbers 
Unemployment = pd.read_excel('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\DataBehandling\\Expected Unemployment.xlsx', sheet_name='Ark1')
Unemployment.index = pd.to_datetime(Unemployment['Date'])
Unemployment.sort_index(inplace=True) # formatting the index
Unemployment['UnemploymentDiff'] = Unemployment['Unemployment'] - Unemployment['Surv(A)']
Unemployment['UnemploymentDummy'] = Unemployment['UnemploymentDiff'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else -1) # News Impact Dummy
Unemployment = Unemployment.loc[: '2023-02-03'] # filtering data wrt. specfied range
Unemployment.to_excel('UnemploymentDummy.xlsx')

# VIX  Data
VIX = yf.Ticker('^VIX')
df = VIX.history(start=start, end=end)
df = pd.DataFrame(df.loc[:,'Close'])
df.rename(columns={'Close':'VIX_d'}, inplace =True) 
# HAR components
df['VIX_1'] = df['VIX_d'].shift(1)
df['VIX_w'] = df['VIX_1'].rolling(5).mean()
df['VIX_w_2'] = df['VIX_1'].rolling(10).mean()
df['VIX_m'] = df['VIX_1'].rolling(22).mean()
df.dropna(inplace=True)

#%% # Descriptive Statistics
"Creating descriptive statistics"
stats_dict = {
    'Number of observations': len(df['VIX_d']),
    'Mean': df['VIX_d'].mean(),
    'Median': df['VIX_d'].median()481 ,
    'Minimum': df['VIX_d'].min(),
    'Maximum': df['VIX_d'].max(),
    'Standard deviation': df['VIX_d'].std(),
    'Skewness': df['VIX_d'].skew(),
    'Kurtosis': df['VIX_d'].kurtosis()
}
print(len(df['VIX_d']))
print(df['VIX_d'].mean())
print(df['VIX_d'].min())
print(df['VIX_d'].max())
print(df['VIX_d'].std())

#%% # Stationarity

"Statistical tests"
# Check for stationarity with adf test

#print(adf(df['VIX_d']))
#print(round(ur.PhillipsPerron(df['VIX_d']).pvalue,8))
print(ur.PhillipsPerron(df['VIX_d']).summary())
print(ur.ADF(df['VIX_d'], method='bic').summary())
#%% # HAR-IS

"HAR In Sample model"
# new dataframe for in sample modelling
HAR_IS = df.copy()
HAR_IS['Date'] = HAR_IS.index
HAR_IS.index = pd.to_datetime(HAR_IS.index)
HAR_IS_results = sm.OLS.from_formula(formula='VIX_d ~ VIX_1+VIX_w+VIX_w_2+VIX_m',
                                      data=HAR_IS).fit(cov_type='HC0')
HAR_IS_params = HAR_IS_results.params # will use these for the out of sample forecast
HAR_IS_resid = HAR_IS_results.resid
HAR_IS_ecdf = sm.distributions.ECDF(HAR_IS_resid)
HAR_IS['predicted']=HAR_IS_results.predict()
HAR_IS_results.summary()


#%% # Variance Inflation Factor-test
#Variance Inflation Factor test
X = HAR_IS[['VIX_1', 'VIX_w', 'VIX_m']]
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

#%% # IV regression

# crating lagges VIX mean series'
HAR_IS = HAR_IS.copy()
HAR_IS['VIX_2'] = HAR_IS['VIX_1'].shift(1)
HAR_IS['VIX_w_1'] = HAR_IS['VIX_w'].shift(1)
HAR_IS['VIX_w_2_1'] = HAR_IS['VIX_w_2'].shift(1)
HAR_IS['VIX_m_1'] = HAR_IS['VIX_m'].shift(1)

# calculating mean values and VIX 2 days before start date 
temp_start = HAR_IS.index[0] - pd.Timedelta(days=40)
temp_end = HAR_IS.index[0]
tempdf = VIX.history(start=temp_start, end=temp_end)
tempdf['VIX_1'] = tempdf['Close'].shift(1)
HAR_IS.loc[HAR_IS.index[0], 'VIX_w_1'] = tempdf.iloc[-5:]['VIX_1'].mean()
HAR_IS.loc[HAR_IS.index[0], 'VIX_w_2_1'] = tempdf.iloc[-10:]['VIX_1'].mean()
HAR_IS.loc[HAR_IS.index[0], 'VIX_m_1'] = tempdf.iloc[-22:]['VIX_1'].mean()
HAR_IS.loc[HAR_IS.index[0], 'VIX_2'] = tempdf.iloc[-1]['VIX_1'] 

# Define the endogenous and exogenous predictor variables
endog = HAR_IS[['VIX_1', 'VIX_w', 'VIX_m']]
exog = [['']]

# Define the instrumental variables for each endogenous predictor variable
ivs = {'VIX_1': 'VIX_2', 'VIX_w' : 'VIX_w_1' , 'VIX_m' : 'VIX_m_1', 'InflationDummy': 'InflationDummy_1',
       'FundsRate': 'FundsRate_1', 'Unemployment': 'Unemployment_1'}

# Create the IV regression formula
formula = 'VIX_d~1+[VIX_1+ VIX_w+VIX_w_2+ VIX_m ~  VIX_2+VIX_w_1+VIX_w_2_1+VIX_m_1]'

# Fit the model using IV2SLS
HAR_IS_results = IV2SLS.from_formula(formula=formula, data=HAR_IS).fit(cov_type='robust')
print(HAR_IS_results)

#%% # GPH Estimator
# Preprocess data to handle missing or infinite values
log_ecdf_y = np.log(HAR_IS_ecdf.y)
log_ecdf_x = np.log(HAR_IS_ecdf.x)

# Remove any rows with missing or infinite values
valid_indices = np.isfinite(log_ecdf_y) & np.isfinite(log_ecdf_x)
log_ecdf_y_valid = log_ecdf_y[valid_indices]
log_ecdf_x_valid = log_ecdf_x[valid_indices]

# Perform the GPH estimator on the valid data
gph_est = sm.OLS(log_ecdf_y_valid, sm.add_constant(log_ecdf_x_valid)).fit()
gph_coeff = gph_est.params[1]  # Get the coefficient estimate
gph_std_err = gph_est.bse[1]  # Get the standard error of the coefficient
gph_t_stat = gph_est.tvalues[1]  # Get the t-statistic
gph_p_value = gph_est.pvalues[1]  # Get the p-value

print(gph_coeff, gph_p_value)

#%% # Philips Estimator

# Preprocess data to handle missing or infinite values
log_ecdf_y = np.log(HAR_IS_ecdf.y)
log_ecdf_x = np.log(HAR_IS_ecdf.x)

# Remove any rows with missing or infinite values
valid_indices = np.isfinite(log_ecdf_y) & np.isfinite(log_ecdf_x)
log_ecdf_y_valid = log_ecdf_y[valid_indices]
log_ecdf_x_valid = log_ecdf_x[valid_indices]

# Perform the Phillips-Perron test on the valid data
phillips_test = ur.PhillipsPerron(df['VIX_d'], trend='c')  # Use lags=1 instead of range(1, 10)
phillips_summary = phillips_test.summary()  # Get the summary of the test results
print(phillips_summary)
#%% # White Test
# Compute White's test for heteroskedasticity
name = ['LM statistic', 'p-value', 
        'f-value', 'f_p-value']
test = sm.stats.diagnostic.het_white(HAR_IS_resid, HAR_IS_results.model.exog)
print(dict(zip(name, test)))

"The output suggests that i do need heteroskedastic-robust standard errors in my model."
"Therefore i make use of the cov_type='HC0' from the statsmodels library"
#%% # Plot HAR-IS
"Plot HAR In Sample model:" 

plt.plot(HAR_IS.index, HAR_IS['predicted'], color='black', alpha =0.8, label='Model')
plt.plot(HAR_IS.index,HAR_IS['VIX_d'], color='blue', alpha =0.5, label='Actual VIX')
plt.ylabel('VIX')
plt.legend()
plt.title('Figure 2: In Sample Model Prediction & Actual VIX')
plt.savefig('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\Grafer\\HAR_in_sample.png')

#%% # Create dfs for HAR-NI & HAR-NIL

"Creating HAR-NI In Sample Model Dataframe"
# Creating Dummy-Variable dataframes
InflationDummy = pd.DataFrame(Inflation[['InflationDummy', 'InflationDiff', 'Date']])
InflationDummy['Date'] = InflationDummy.index

FundsRateDummy = pd.DataFrame(FundsRate[['FundsRateDummy', 'FundsRateDiff', 'Date']])
FundsRateDummy['Date'] = FundsRateDummy.index

UnemploymentDummy = pd.DataFrame(Unemployment[['UnemploymentDummy','UnemploymentDiff', 'Date']])
UnemploymentDummy['Date'] = UnemploymentDummy.index

# Creating Dataframe containing HAR components
tempdf1 = df.copy()
tempdf1.index = tempdf1.index.date
tempdf1['Date'] = tempdf1.index

# Adding dummy variables to the model
tempdf2 = pd.concat([InflationDummy.set_index('Date'), 
                     FundsRateDummy.set_index('Date'), 
                     UnemploymentDummy.set_index('Date'), 
                     tempdf1.set_index('Date')], 
                     axis=1, join='outer')

# Sort the resulting DataFrame based on the 'Date' column
tempdf2 = tempdf2.sort_index()
tempdf2['Date'] = tempdf2.index

# Fill in the missing values in the Dummy columns using forward fill
tempdf2['InflationDummy'] = tempdf2['InflationDummy'].fillna(method='ffill')
tempdf2['InflationDiff'] = tempdf2['InflationDiff'].fillna(method='ffill')
tempdf2['FundsRateDummy'] = tempdf2['FundsRateDummy'].fillna(method='ffill')
tempdf2['FundsRateDiff'] = tempdf2['FundsRateDiff'].fillna(method='ffill')
tempdf2['UnemploymentDummy'] = tempdf2['UnemploymentDummy'].fillna(method='ffill')
tempdf2['UnemploymentDiff'] = tempdf2['UnemploymentDiff'].fillna(method='ffill')

# Filling in VIX nans with duplicates that we will drop after
tempdf2['VIX_d'] = tempdf2['VIX_d'].fillna(method='ffill')
tempdf2['VIX_1'] = tempdf2['VIX_1'].fillna(method='ffill')
tempdf2['VIX_w'] = tempdf2['VIX_w'].fillna(method='ffill')
tempdf2['VIX_m'] = tempdf2['VIX_m'].fillna(method='ffill')

# Drop duplicates
tempdf2.drop_duplicates(subset=['Date'],keep='last', inplace=True)
for date in tempdf2['Date']:
    if str(date)[:10] not in df.index:
        tempdf2 = tempdf2[tempdf2['Date'] != date]
tempdf2.loc[tempdf2['Date'] == '2022-12-13', 'VIX_d'] = df.loc['2022-12-13', 'VIX_d']
tempdf2.loc[tempdf2['Date'] == '2022-12-13', 'VIX_1'] = df.loc['2022-12-13', 'VIX_1']
tempdf2.loc[tempdf2['Date'] == '2022-12-13', 'VIX_w'] = df.loc['2022-12-13', 'VIX_w']
tempdf2.loc[tempdf2['Date'] == '2022-12-13', 'VIX_m'] = df.loc['2022-12-13', 'VIX_m']

# Defing model without NI levels
HAR_NI_IS = tempdf2.copy()
HAR_NI_IS.drop(labels= ['InflationDiff', 'FundsRateDiff', 'UnemploymentDiff'], axis='columns', inplace=True)

# Defining model with NI levels
HAR_NIL_IS = tempdf2.copy()
HAR_NIL_IS.set_index('Date', inplace=True)

#%% #HAR-NI
"Create the model for the In Sample HAR News Impact Model without Levels"

HAR_NI_IS_results = sm.OLS.from_formula(formula='VIX_d ~ VIX_1+VIX_w+VIX_m+InflationDummy+FundsRateDummy+UnemploymentDummy'
                              , data=HAR_NI_IS).fit(cov_type='HC0')

HAR_NI_IS_params = HAR_NI_IS_results.params
HAR_NI_IS_resid = HAR_NI_IS_results.resid
HAR_NI_IS_ecdf = sm.distributions.ECDF(HAR_NI_IS_resid)
HAR_NI_IS['predicted']=HAR_NI_IS_results.predict()
HAR_NI_IS_results.summary()
HAR_NI_IS_cov_matrix = HAR_NI_IS_results.cov_params()
print(HAR_NI_IS_cov_matrix)
#%%
#Variance Inflation Factor test
X = HAR_NI_IS[['InflationDummy', 
                'FundsRateDummy', 
                'UnemploymentDummy']]
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
#%% # White Test
# Compute White's test for heteroskedasticity
name = ['LM statistic', 'p-value', 
        'f-value', 'f_p-value']
test = sm.stats.diagnostic.het_white(HAR_NI_IS_resid, HAR_NI_IS_results.model.exog)
print(dict(zip(name, test)))

"The output suggests that i do need heteroskedastic-robust standard errors in my model."
"Therefore i make use of the cov_type='HC0' from the statsmodels library"

#%% # HAR-NIL
"I create the model for the In Sample HAR News Impact Model with Levels"

HAR_NIL_IS_results = sm.OLS.from_formula(formula='VIX_d ~ VIX_1+VIX_w+VIX_m+InflationDummy+InflationDiff+FundsRateDummy+FundsRateDiff+UnemploymentDummy+UnemploymentDiff'
                              , data=HAR_NIL_IS).fit(cov_type='HC0')
HAR_NIL_IS_params = HAR_NIL_IS_results.params # will use these for the out of sample forecast
HAR_NIL_IS_resid = HAR_NIL_IS_results.resid
HAR_NIL_IS_scale = HAR_NIL_IS_results.scale
HAR_NIL_IS['predicted']=HAR_NIL_IS_results.predict()
HAR_NIL_IS_results.summary()
HAR_NIL_IS_cov_matrix = HAR_NIL_IS_results.cov_params()
print(HAR_NIL_IS_cov_matrix)
#%% #Variance Inflation Factor test
X = HAR_NIL_IS[['VIX_1', 'VIX_w', 'VIX_m', 
                'InflationDummy', 'InflationDiff', 
                'FundsRateDummy', 'FundsRateDiff', 
                'UnemploymentDummy', 'UnemploymentDiff']]
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

#%% # IV regression for adressing Multicolinearity

# crating lagges VIX mean series'
HAR_NIL_IV = HAR_NIL_IS.copy()
HAR_NIL_IV['VIX_2'] = HAR_NIL_IV['VIX_1'].shift(1)
HAR_NIL_IV['VIX_w_1'] = HAR_NIL_IV['VIX_w'].shift(1)
HAR_NIL_IV['VIX_m_1'] = HAR_NIL_IV['VIX_m'].shift(1)

# calculating mean values and VIX 2 days before start date 
temp_start = HAR_NIL_IV.index[0] - pd.Timedelta(days=40)
temp_end = HAR_NIL_IV.index[0]
tempdf = VIX.history(start=temp_start, end=temp_end)
tempdf['VIX_1'] = tempdf['Close'].shift(1)
HAR_NIL_IV.loc[HAR_NIL_IV.index[0], 'VIX_w_1'] = tempdf.iloc[-5:]['VIX_1'].mean()
HAR_NIL_IV.loc[HAR_NIL_IV.index[0], 'VIX_m_1'] = tempdf.iloc[-22:]['VIX_1'].mean()
HAR_NIL_IV.loc[HAR_NIL_IV.index[0], 'VIX_2'] = tempdf.iloc[-1]['VIX_1']

# creating lagged dummy series'
HAR_NIL_IV['InflationDummy_1'] = HAR_NIL_IV['InflationDummy'].shift(1)
HAR_NIL_IV['InflationDummy_1'] = HAR_NIL_IV['InflationDummy_1'].fillna(method='bfill')
HAR_NIL_IV['FundsRateDummy_1'] = HAR_NIL_IV['FundsRateDummy'].shift(1)
HAR_NIL_IV['FundsRateDummy_1'] = HAR_NIL_IV['FundsRateDummy_1'].fillna(method='bfill') 
HAR_NIL_IV['UnemploymentDummy_1'] = HAR_NIL_IV['UnemploymentDummy'].shift(1) 
HAR_NIL_IV['UnemploymentDummy_1'] = HAR_NIL_IV['UnemploymentDummy_1'].fillna(method='bfill') 

# Define the endogenous and exogenous predictor variables
endog = HAR_NIL_IV[['VIX_1', 'VIX_w', 'VIX_m', 'InflationDummy', 'FundsRateDummy', 'UnemploymentDummy']]
exog = HAR_NIL_IV[['InflationDiff', 'FundsRateDiff', 'UnemploymentDiff']]

# Define the instrumental variables for each endogenous predictor variable
ivs = {'VIX_1': 'VIX_2', 'VIX_w' : 'VIX_w_1' , 'VIX_m' : 'VIX_m_1', 'InflationDummy': 'InflationDummy_1',
       'FundsRateDummy': 'FundsRateDummy_1', 'UnemploymentDummy': 'UnemploymentDummy_1'}

# Create the IV regression formula
formula = 'VIX_d~1+InflationDiff+FundsRateDiff+UnemploymentDiff + [VIX_1+ VIX_w+ VIX_m + InflationDummy + FundsRateDummy + UnemploymentDummy ~ VIX_2 + VIX_w_1+ VIX_m_1 + InflationDummy_1 + FundsRateDummy_1 + UnemploymentDummy_1]'

# Fit the model using IV2SLS
HAR_NIL_IV_results = IV2SLS.from_formula(formula=formula, data=HAR_NIL_IV).fit(cov_type='robust')
print(HAR_NIL_IV_results)

#%% # Plot HAR-NI & HAR-NIL

"Plot HAR News Impact Models" 

plt.plot(HAR_NI_IS.index, HAR_NI_IS['predicted'], color='black', alpha =0.8, label='HAR-NI Model')
plt.plot(HAR_NIL_IS.index, HAR_NIL_IS['predicted'], color='red', alpha =0.8, label='HAR-NIL Model')
plt.plot(HAR_NI_IS.index,HAR_NI_IS['VIX_d'], color='blue', alpha =0.5, label='Actual VIX')
plt.ylabel('VIX')
plt.legend()
plt.title('Figure 2: In Sample Model Prediction & Actual VIX')
plt.savefig('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\Grafer\\HAR_NI_in_sample.png')

#%% # Create df for HAR-MN
"HAR- Macro News In Sample model"

# Creating Macro Statistic dataframes
InflationMN = pd.DataFrame(Inflation[['Inflation', 'Date']])
InflationMN['Date'] = InflationMN.index

FundsRateMN = pd.DataFrame(FundsRate[['FundsRate', 'Date']])
FundsRateMN['Date'] = FundsRateMN.index

UnemploymentMN = pd.DataFrame(Unemployment[['Unemployment', 'Date']])
UnemploymentMN['Date'] = UnemploymentMN.index

# Creating Dataframe containing HAR components
tempdf3 = df.copy()
tempdf3.index = tempdf3.index.date
tempdf3['Date'] = tempdf3.index

# Adding dummy variables to the model
tempdf4 = pd.concat([InflationMN.set_index('Date'), 
                     FundsRateMN.set_index('Date'), 
                     UnemploymentMN.set_index('Date'), 
                     tempdf3.set_index('Date')], 
                     axis=1, join='outer')

# Sort the resulting DataFrame based on the 'Date' column
tempdf4 = tempdf4.sort_index()
tempdf4['Date'] = tempdf4.index

# Fill in the missing values in the Macro Statistic columns using forward fill
tempdf4['Inflation'] = tempdf4['Inflation'].fillna(method='ffill')

tempdf4['FundsRate'] = tempdf4['FundsRate'].fillna(method='ffill')

tempdf4['Unemployment'] = tempdf4['Unemployment'].fillna(method='ffill')

# Filling in VIX nans with duplicates that we will drop after
tempdf4['VIX_d'] = tempdf4['VIX_d'].fillna(method='ffill')
tempdf4['VIX_1'] = tempdf4['VIX_1'].fillna(method='ffill')
tempdf4['VIX_w'] = tempdf4['VIX_w'].fillna(method='ffill')
tempdf4['VIX_m'] = tempdf4['VIX_m'].fillna(method='ffill')

# Drop duplicates
tempdf4.drop_duplicates(subset=['Date'],keep='last', inplace=True)
for date in tempdf4['Date']:
    if str(date)[:10] not in df.index:
        tempdf4 = tempdf4[tempdf4['Date'] != date]
tempdf4.loc[tempdf4['Date'] == '2022-12-13', 'VIX_d'] = df.loc['2022-12-13', 'VIX_d']
tempdf4.loc[tempdf4['Date'] == '2022-12-13', 'VIX_1'] = df.loc['2022-12-13', 'VIX_1']
tempdf4.loc[tempdf4['Date'] == '2022-12-13', 'VIX_w'] = df.loc['2022-12-13', 'VIX_w']
tempdf4.loc[tempdf4['Date'] == '2022-12-13', 'VIX_m'] = df.loc['2022-12-13', 'VIX_m']


#%% # HAR-MN
"HAR- Macro News In Sample model"
HAR_MS_IS = tempdf4.copy()
HAR_MS_IS.set_index('Date', inplace=True)
HAR_MS_IS_results = sm.OLS.from_formula(formula='VIX_d ~ VIX_1+VIX_w+VIX_m+Inflation+FundsRate+Unemployment',
                                      data=HAR_MS_IS).fit(cov_type='HC0')
HAR_MS_IS_params = HAR_MS_IS_results.params # will use these for the out of sample forecast
HAR_MS_IS_resid = HAR_MS_IS_results.resid
HAR_MS_IS['predicted']=HAR_MS_IS_results.predict()
HAR_MS_IS_results.summary()

#Variance Inflation Factor test
X = HAR_MS_IS[['VIX_1', 'VIX_w', 'VIX_m', 
                'Inflation', 
                'FundsRate', 
                'Unemployment']]
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
#%% #IV regression for adressing Multicolinearity

# crating lagges VIX mean series'
HAR_MS_IV = HAR_MS_IS.copy()
HAR_MS_IV['VIX_2'] = HAR_MS_IV['VIX_1'].shift(1)
HAR_MS_IV['VIX_w_1'] = HAR_MS_IV['VIX_w'].shift(1)
HAR_MS_IV['VIX_m_1'] = HAR_MS_IV['VIX_m'].shift(1)

# calculating mean values and VIX 2 days before start date 
temp_start = HAR_MS_IV.index[0] - pd.Timedelta(days=40)
temp_end = HAR_MS_IV.index[0]
tempdf = VIX.history(start=temp_start, end=temp_end)
tempdf['VIX_1'] = tempdf['Close'].shift(1)
HAR_MS_IV.loc[HAR_MS_IV.index[0], 'VIX_w_1'] = tempdf.iloc[-5:]['VIX_1'].mean()
HAR_MS_IV.loc[HAR_MS_IV.index[0], 'VIX_m_1'] = tempdf.iloc[-22:]['VIX_1'].mean()
HAR_MS_IV.loc[HAR_MS_IV.index[0], 'VIX_2'] = tempdf.iloc[-1]['VIX_1']

# creating lagged dummy series'
HAR_MS_IV['Inflation_1'] = HAR_MS_IV['Inflation'].shift(1)
HAR_MS_IV['Inflation_1'] = HAR_MS_IV['Inflation_1'].fillna(method='bfill')
HAR_MS_IV['FundsRate_1'] = HAR_MS_IV['FundsRate'].shift(1)
HAR_MS_IV['FundsRate_1'] = HAR_MS_IV['FundsRate_1'].fillna(method='bfill') 
HAR_MS_IV['Unemployment_1'] = HAR_MS_IV['Unemployment'].shift(1) 
HAR_MS_IV['Unemployment_1'] = HAR_MS_IV['Unemployment_1'].fillna(method='bfill') 

# Define the endogenous and exogenous predictor variables
endog = HAR_MS_IV[['VIX_1', 'VIX_w', 'VIX_m', 'Inflation', 'FundsRate', 'Unemployment']]
exog = [['']]

# Define the instrumental variables for each endogenous predictor variable
ivs = {'VIX_1': 'VIX_2', 'VIX_w' : 'VIX_w_1' , 'VIX_m' : 'VIX_m_1', 'InflationDummy': 'InflationDummy_1',
       'FundsRate': 'FundsRate_1', 'Unemployment': 'Unemployment_1'}

# Create the IV regression formula
formula = 'VIX_d~1+[VIX_1+ VIX_w+ VIX_m+Inflation + FundsRate + Unemployment ~  VIX_2+VIX_w_1+VIX_m_1+Inflation_1 + FundsRate_1 + Unemployment_1]'

# Fit the model using IV2SLS
HAR_MS_IV_results = IV2SLS.from_formula(formula=formula, data=HAR_MS_IV).fit(cov_type='robust')
print(HAR_MS_IV_results)
#%%
# Comparing all 4 models - F-test of nested models

# Calculate the residual sum of squares (RSS) for each model
RSS_har = HAR_IS_results.ssr
RSS_har_ni = HAR_NI_IS_results.ssr
RSS_har_nil = HAR_NIL_IS_results.ssr
RSS_har_ms = HAR_MS_IS_results.ssr

# Calculate the degrees of freedom difference for each model
df_diff_har_ni = HAR_IS_results.df_resid - HAR_NI_IS_results.df_resid
df_diff_har_nil = HAR_IS_results.df_resid - HAR_NIL_IS_results.df_resid
df_diff_har_ms = HAR_IS_results.df_resid - HAR_MS_IS_results.df_resid

# Calculate the F-statistic
F_har_ni = ((RSS_har - RSS_har_ni) / df_diff_har_ni) / (RSS_har_ni / HAR_NI_IS_results.df_resid)
F_har_nil = ((RSS_har - RSS_har_nil) / df_diff_har_nil) / (RSS_har_nil / HAR_NIL_IS_results.df_resid)
F_har_ms = ((RSS_har - RSS_har_ms) / df_diff_har_ms) / (RSS_har_ms / HAR_MS_IS_results.df_resid)

# Calculate the p-values using the F-distribution
p_value_har_ni = 1 - stats.f.cdf(F_har_ni, df_diff_har_ni, HAR_NI_IS_results.df_resid)
p_value_har_nil = 1 - stats.f.cdf(F_har_nil, df_diff_har_nil, HAR_NIL_IS_results.df_resid)
p_value_har_ms = 1 - stats.f.cdf(F_har_ms, df_diff_har_ms, HAR_MS_IS_results.df_resid)

print(F_har_ni, p_value_har_ni, df_diff_har_ni)
print(F_har_nil, p_value_har_nil, df_diff_har_nil)
print(F_har_ms, p_value_har_ms, df_diff_har_ms)
#%% # Setup for HAR-OS forecast loop
"Out of sample model"

"we begin by setting up the initial dataframe for the forecast loop"
# initialize dataframe
HAR_OS_s = HAR_IS.iloc[-1:]

# add last 22 rows of VIX_1 to starting dataframe in descending order for mean calcs
last_22_rows = HAR_IS['VIX_1'].tail(22).sort_index(ascending=False)
for index, value in last_22_rows.items():
    HAR_OS_s.loc[index, 'VIX_1'] = value
HAR_OS_s = HAR_OS_s.sort_index(ascending=True)

# add empty row with next business day as index
next_business_day = pd.date_range(start=HAR_OS_s.index[-1], periods=2, freq='B')[1]
HAR_OS_s.loc[next_business_day] = ['nan'] * len(HAR_OS_s.columns)

# we drop the VIX column as it is irellevant in our forecast
HAR_OS_s.drop('VIX_d', axis=1, inplace=True)

# today, VIX lagged is equal to VIX yesterday
HAR_OS_s.loc[HAR_OS_s.index[-1], 'VIX_1'] = HAR_IS.iloc[-1]['VIX_d']

# create our mean values used in OLS regression
HAR_OS_s.loc[HAR_OS_s.index[-1], 'VIX_w'] = HAR_OS_s.iloc[-5:]['VIX_1'].mean()
HAR_OS_s.loc[HAR_OS_s.index[-1], 'VIX_m']  = HAR_OS_s.iloc[-22:]['VIX_1'].mean()

# create dataframe containing the last row which I use as regressors
row = HAR_OS_s.iloc[-1]
row.drop('predicted', axis=0, inplace=True)

# calculate OLS and add OLS prediction HAR_OS_s['predicted'] column
OLS = HAR_IS_params[0] + np.dot(row.squeeze().T, HAR_IS_params[1:]) + HAR_IS_resid.mean()
HAR_OS_s.loc[HAR_OS_s.index[-1],'predicted']= OLS

#%% # 60 day HAR-OS forecast loop starting today

"loop for creating forecast"

# add empty row with next business day as index
next_business_day = pd.date_range(start=HAR_OS_s.index[-1], periods=2, freq='B')[1]
HAR_OS_s.loc[next_business_day] = [None] * len(HAR_OS_s.columns)

for i in range(60):
    # prediction at time t-1 is now VIX_1 at time t
    HAR_OS_s.loc[HAR_OS_s.index[-1],'VIX_1'] = HAR_OS_s.iloc[-2]['predicted'] 

    # create our mean values used in OLS regression
    HAR_OS_s.loc[HAR_OS_s.index[-1],'VIX_w'] = HAR_OS_s.iloc[-5:]['VIX_1'].mean()
    HAR_OS_s.loc[HAR_OS_s.index[-1],'VIX_m'] = HAR_OS_s.iloc[-22:]['VIX_1'].mean()

    # create dataframe containing the last row which I use as regressors
    row = HAR_OS_s.iloc[-1]
    row.drop('predicted', axis=0, inplace=True)

    # calculate OLS and add OLS prediction HAR_OS_s['predicted'] column
    OLS = HAR_IS_params[0] + np.dot(row.squeeze().T, HAR_IS_params[1:]) + HAR_IS_resid.mean()
    HAR_OS_s.loc[HAR_OS_s.index[-1],'predicted']= OLS
    
    # add empty row with next business day as index
    next_business_day = pd.date_range(start=HAR_OS_s.index[-1], periods=2, freq='B')[1]
    HAR_OS_s.loc[next_business_day] = [None] * len(HAR_OS_s.columns)
    
    # stop loop if 60 business days have been added
    if i == 59:
        break

#%% # plot 60 day HAR-OS forecast

# create a clean out of sample dataframe with predicted values
HAR_OS = HAR_OS_s.dropna(subset=['predicted'])
HAR_OS = HAR_OS.iloc[1:]

# If the VIX forecast end is < today, we compare with actual VIX data
VIX_forecast_start = HAR_OS.index.min()
VIX_forecast_end = HAR_OS.index.max() + pd.tseries.offsets.BDay()
VIX_forecast_compare = VIX.history(start=VIX_forecast_start, end=VIX_forecast_end)

VIX_forecast_compare = pd.DataFrame(VIX_forecast_compare.loc[:,'Close'])
VIX_forecast_compare.rename(columns={'Close':'VIX_d'}, inplace =True) # rename the close column

# plot the predicted values
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(HAR_OS.index, HAR_OS['predicted'], label='Forecast')
ax.plot(VIX_forecast_compare.index, VIX_forecast_compare['VIX_d'], label='VIX')
ax.set_title('Figure 4: 60 day VIX forecast')
ax.legend()
plt.savefig('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\Grafer\\vix_HAR_OS_forecast.png')
#%% # plot HAR-IS & HAR-OS forecast with actual VIX

# create dataframe for full dataset comparison
HAR_Forecast_All = pd.concat([HAR_IS, HAR_OS], axis=0, ignore_index=True, join='outer')

HAR_IS['Date'] = HAR_IS.index
HAR_OS['Date'] = HAR_OS.index
HAR_Forecast_All.index = pd.concat([HAR_IS['Date'], HAR_OS['Date']], axis=0)
HAR_Forecast_All['VIX_d'] = pd.concat([HAR_IS['VIX_d'], VIX_forecast_compare['VIX_d']], axis=0, join='outer')

# plot both historical and forecast
figs, axs = plt.subplots(figsize=(10,7))
axs.plot(HAR_Forecast_All['Date'], HAR_Forecast_All['predicted'], label='Forecast')
axs.plot(HAR_Forecast_All['Date'], HAR_Forecast_All['VIX_d'], label='VIX')
axs.set_title('In Sample and Out of Sample Forecast')
axs.legend()
plt.savefig('C:\\Users\\click\\OneDrive - University of Copenhagen\\Documents\\BA Projekt\\Grafer\\vix_HAR_forecast_ALL.png')



#%% Portoflio Section DES
"""
In this following Section i will :
1. Create a portfolio consisting of the S&P500 and VIXY ETF
2. Weights will be chosen on the basis of MVP
3. Create a measure of the probability of a VIX increase based on Granger and Perasan method 
4. Calculate expected utility for long and short positions based on probability measure
5. Take long and short positions each day based on the above utility estimates
6. Calculate annualized returns based on postioning choices in this portfolio
"""
#%% # Portfolio Weight Construction 

pf_start = datetime(2017,1,1)
pf_end = datetime(2023,2,1)

# VIXY Data
VIXY_tick = yf.Ticker('VIXY')
VIXY = VIXY_tick.history(start=pf_start, end=pf_end)
VIXY = pd.DataFrame(VIXY.loc[:, 'Close'])
VIXY.rename(columns={'Close':'VIXY'}, inplace =True) 

# S&P Data
SP_tick = yf.Ticker('SPY')
SP = SP_tick.history(start=pf_start, end=pf_end)
SP = pd.DataFrame(SP.loc[:, 'Close'])
SP.rename(columns={'Close':'S&P'}, inplace =True) 

# Concatenate VIXY and SP data into a single dataframe
VIXY_SP_Close = pd.concat([VIXY, SP], axis=1)
VIXY_SP_Close = VIXY_SP_Close.loc['2018-02-02':]

# gauging the data - big invese correlation
fig, axp = plt.subplots(figsize=(10,7))
axp.plot(VIXY_SP_Close.index, VIXY_SP_Close['S&P'], color='blue', label='S&P 500')
axp.plot(VIXY_SP_Close.index, VIXY_SP_Close['VIXY'], color='black',label='VIXY')
axp.set_title('VIXY and S&P')
axp.legend()

# Rebalancing MVP weights each quarter
delta = relativedelta(pf_end, pf_start)
num_months = delta.years * 12 + delta.months
monthly_weights = pd.DataFrame(columns=['VIXY', 'S&P'])
pf_weight_data = pd.concat([VIXY, SP], axis=1)

# Loop through each quarter in portfolio holding period and rebalance
for i in range(0, num_months):
    end_date = pf_start + pd.DateOffset(months=i) - pd.DateOffset(days=1)
    data_up_to_month_end = pf_weight_data.loc[:end_date]
    returns = data_up_to_month_end.pct_change().dropna()
    cov = np.cov(returns.T)
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(cov))
    monthly_weights = np.dot(inv_cov, ones) / np.dot(np.dot(ones, inv_cov), ones)
    monthly_weights.loc[end_date] = monthly_weights
    monthly_weights = monthly_weights.append({'Month': end_date,
                                               'VIXY': monthly_weights['VIXY'],
                                               'S&P': monthly_weights['S&P']})
                                              
# Forward fill to get weights for all dates
VIXY_SP_Close.index = VIXY_SP_Close.index.tz_localize(None)
monthly_weights.set_index('Month', inplace=True)
#%%
pf_weights = monthly_weights.reindex(VIXY_SP_Close.index, method='ffill')
pf_weights = pf_weights.rename(columns={'VIXY': 'VIXY_weight', 'S&P': 'SP_weight'})

#%% # PF stuff
# Index returns
VIXY_SP_Returns = pd.concat([VIXY, SP], axis=1)
VIXY_SP_Returns['VIXY_return'] = VIXY_SP_Returns['VIXY'].pct_change()
VIXY_SP_Returns['S&P_return'] = VIXY_SP_Returns['S&P'].pct_change()
VIXY_SP_Returns.drop(['VIXY', 'S&P'], axis=1, inplace=True)
VIXY_SP_Returns.index = VIXY_SP_Returns.index.tz_localize(pf_weights.index.tz) 

# Calculate portfolio returns
pf_returns = pd.concat([pf_weights, VIXY_SP_Returns], axis=1, join='outer')
pf_returns['pf_return'] = pf_returns.apply(lambda x: np.dot(x[0:2], x[2:]), axis=1)
pf_returns = pf_returns.loc['2018-02-02':]
pf_returns['Index'] = 100
for i in range(1, len(pf_returns)):
    pf_returns['Index'][i] = pf_returns['Index'][i-1] * (1 + pf_returns['pf_return'][i])



fig, axt = plt.subplots(figsize=(10,7))
axx = axt.twinx()
axx.plot(pf_returns.index, pf_returns['Index'], color='red', label='PF returns')
axt.plot(VIXY_SP_Close.index, VIXY_SP_Close['S&P'], color='blue', label='S&P 500')
axt.plot(VIXY_SP_Close.index, VIXY_SP_Close['VIXY'], color='black',label='VIXY')
axp.set_title('VIXY and S&P')
axp.legend()


#%% # Average log returns of short/long position on days where VIX/S&P rose/fell

# Create VIXY_SP_Returns_short
VIXY_SP_Returns_short = VIXY_SP_Returns.copy()
VIXY_SP_Returns_short.loc[VIXY_SP_Returns.index, 'VIXY_return'] *= -1
VIXY_SP_Returns_short.loc[VIXY_SP_Returns.index, 'S&P_return'] *= -1

pf_start = pd.to_datetime('2017-01-01')
pf_end = pd.to_datetime('2023-02-01')
#num_months = (pf_end- pf_start) // pd.Timedelta('1M')   # 73 months

returns_summary = pd.DataFrame(columns=['Month', 'VIXY_Positive_days', 'VIXY_Negative_days',
                                        'Long_LogR_VIXY_positive', 'Long_LogR_VIXY_negative',
                                        'SP_Positive_days', 'SP_Negative_days',
                                        'Long_LogR_SP_positive', 'Long_LogR_SP_negative',
                                        'Short_LogR_VIXY_positive', 'Short_LogR_VIXY_negative',
                                        'Short_LogR_SP_positive', 'Short_LogR_SP_negative'])

# Loop through each month in portfolio holding period
for i in range(73):
    
    month_start_date = pf_start + pd.DateOffset(months=i)
    month_end_date = month_start_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    month_long_data = VIXY_SP_Returns.loc[month_start_date:month_end_date]
    month_short_data = VIXY_SP_Returns_short.loc[month_start_date:month_end_date]

    # Calculate the number of positive and negative return days and average log returns for long position
    VIXY_positive_returns = month_long_data['VIXY_return'][month_long_data['VIXY_return'] > 0]
    VIXY_negative_returns = month_long_data['VIXY_return'][month_long_data['VIXY_return'] < 0]
    VIXY_positive_log_returns = np.mean(np.log(1 + VIXY_positive_returns))
    VIXY_negative_log_returns = np.mean(np.log(1 + VIXY_negative_returns))
    
    SP_positive_returns = month_long_data['S&P_return'][month_long_data['S&P_return'] > 0]
    SP_negative_returns = month_long_data['S&P_return'][month_long_data['S&P_return'] < 0]
    SP_positive_log_returns = np.mean(np.log(1 + SP_positive_returns))
    SP_negative_log_returns = np.mean(np.log(1 + SP_negative_returns))

    # Calculate the average log returns when shorting VIXY/SPY on days they fell/rose respectively
    Short_VIXY_positive_log_returns = np.mean(np.log(1 - VIXY_positive_returns))
    Short_VIXY_negative_log_returns = np.mean(np.log(1 - VIXY_negative_returns))
    Short_SP_positive_log_returns = np.mean(np.log(1 - SP_positive_returns))
    Short_SP_negative_log_returns = np.mean(np.log(1 - SP_negative_returns))

    # Append the results to returns_summary dataframe
    returns_summary = returns_summary.append({'Month': month_end_date, 
                                           'VIXY_Positive_days': len(VIXY_positive_returns), 
                                           'VIXY_Negative_days': len(VIXY_negative_returns), 
                                           'Long_LogR_VIXY_positive': VIXY_positive_log_returns, 
                                           'Long_LogR_VIXY_negative': VIXY_negative_log_returns, 
                                           'SP_Positive_days': len(SP_positive_returns), 
                                           'SP_Negative_days': len(SP_negative_returns), 
                                           'Long_LogR_SP_positive': SP_positive_log_returns, 
                                           'Long_LogR_SP_negative': SP_negative_log_returns, 
                                           'Short_LogR_VIXY_positive': Short_VIXY_positive_log_returns, 
                                           'Short_LogR_VIXY_negative': Short_VIXY_negative_log_returns,  
                                           'Short_LogR_SP_positive': Short_SP_positive_log_returns, 
                                           'Short_LogR_SP_negative': Short_SP_negative_log_returns}, 
                                          ignore_index=True)

# Set Month column as index
returns_summary.set_index('Month', inplace=True)
returns_summary.to_excel('returns_summary.xlsx')
#Short_SP_negative_returns.to_excel('test.xlsx')
