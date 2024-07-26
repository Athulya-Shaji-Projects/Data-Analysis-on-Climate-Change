import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
# %matplotlib inline
import plotly.offline as py

#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
#reading csv and cleaning the data, dropping unwanted columns

global_temp_country = pd.read_csv('C:/Users/DELL/Downloads/TempChange/GlobalLandTemperaturesByCountry.csv')
global_temp = pd.read_csv('C:/Users/DELL/Downloads/TempChange/GlobalLandTemperaturesMean.csv')

global_temp_country = global_temp_country[global_temp_country['AverageTemperature'].notna()]
global_temp['dt'] = pd.to_datetime(global_temp['dt'])
countries = np.unique(global_temp_country['Country'])

mean_temp = []
for country in countries:
    mean_temp.append(global_temp_country[global_temp_country['Country'] ==
                                               country]['AverageTemperature'].mean())
global_temp = global_temp.reset_index()
global_temp['Country'] = countries
global_temp['Temp'] = mean_temp
global_temp = global_temp[['Country','Temp']]

print(global_temp.head())

forest_df = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/Forest Area.csv')
forest_df = forest_df[['Country', 'Forest Area']]
forest_df = forest_df[forest_df['Forest Area'].notna()]
print(forest_df.head())

climate_disaster_df = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/Climatological disasters.csv')
climate_disaster_df = climate_disaster_df[['Country', 'Climatic Disaster']]
climate_disaster_df = climate_disaster_df[climate_disaster_df['Climatic Disaster'].notna()]
print(climate_disaster_df.head())

CH4_df = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/CH4_Emissions.csv')
CH4_df = CH4_df[['Country', 'TotalCH4']]
CH4_df = CH4_df[CH4_df['TotalCH4'].notna()]
print(CH4_df.head())

CO2_df = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/CO2_Emissions.csv')
CO2_df = CO2_df[['Country', 'TotalCO2']]
CO2_df = CO2_df[CO2_df['TotalCO2'].notna()]
print(CO2_df.head())

GHG_df = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/GHG_Emissions.csv')
GHG_df = GHG_df[['Country', 'TotalGHG']]
GHG_df = GHG_df[GHG_df['TotalGHG'].notna()]
print(GHG_df.head())

N2O_df = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/N2O_Emissions.csv')
N2O_df = N2O_df[['Country', 'TotalN2O']]
N2O_df = N2O_df[N2O_df['TotalN2O'].notna()]
print(N2O_df.head())

NOx_df = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/NOx_Emissions.csv')
NOx_df = NOx_df[['Country', 'TotalNOx']]
NOx_df = NOx_df[NOx_df['TotalNOx'].notna()]
print(NOx_df.head())

SO2_df = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/SO2_emissions.csv')
SO2_df = SO2_df[['Country', 'TotalSO2']]
SO2_df = SO2_df[SO2_df['TotalSO2'].notna()]
print(SO2_df.head())

#mearginh different data together
Air_df = pd.merge(CO2_df,CH4_df,on='Country',how='inner')
Air_df = pd.merge(Air_df,NOx_df,on='Country',how='inner')
Air_df = pd.merge(Air_df,N2O_df,on='Country',how='inner')
Air_df = pd.merge(Air_df,SO2_df,on='Country',how='inner')
Air_df = pd.merge(Air_df,GHG_df,on='Country',how='inner')
#Air_df = pd.merge(Air_df,global_temp,on='Country',how='inner')
#Air_df = pd.merge(Air_df,forest_df,on='Country',how='inner')
#Air_df = pd.merge(Air_df,climate_disaster_df,on='Country',how='inner')

print(Air_df.head())

#plotting correlation
corr = Air_df.corr()
print(corr)
sns.heatmap(corr)
plt.show()

#training  uning linear reg
X = Air_df['TotalCO2'].values.reshape(-1,1)
y = Air_df['TotalGHG'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg = LinearRegression()
reg.fit(X_train, y_train)

print('Intercept',reg.intercept_)

print('slope',reg.coef_)
y_pred = reg.predict(X_test)

#accuracy score
score=r2_score(y_test,y_pred)
print('r2 socre ',score)
print('mean_sqrd_error',mean_squared_error(y_test,y_pred))
print('root_mean_squared error',np.sqrt(mean_squared_error(y_test,y_pred)))


#plotying actual and predicted value from the model
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())
plt.plot(df)
plt.show()



#training  uning linear reg
X = Air_df['TotalCO2'].values.reshape(-1,1)
y = Air_df['TotalNOx'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg = LinearRegression()
reg.fit(X_train, y_train)

print('Intercept',reg.intercept_)

print('slope',reg.coef_)
y_pred = reg.predict(X_test)

#accuracy score
score=r2_score(y_test,y_pred)
print('r2 socre ',score)
print('mean_sqrd_error',mean_squared_error(y_test,y_pred))
print('root_mean_squared error',np.sqrt(mean_squared_error(y_test,y_pred)))

#plotying actual and predicted value from the model
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())
plt.plot(df)
plt.show()


