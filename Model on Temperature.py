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
#reading csv
global_temp_country = pd.read_csv('C:/Users/DELL/Downloads/TempChange/GlobalLandTemperaturesByCountry.csv')
global_temp = pd.read_csv('C:/Users/DELL/Downloads/TempChange/GlobalLandTemperaturesMean.csv')

#cleaning data
global_temp_country = global_temp_country[global_temp_country['AverageTemperature'].notna()]
global_temp['dt'] = pd.to_datetime(global_temp['dt'])
countries = np.unique(global_temp_country['Country'])

#finding avg, min and max temp of countires
mean_temp = []
max_temp_list = []
min_temp_list = []
for country in countries:
    mean_temp.append(global_temp_country[global_temp_country['Country'] ==
                                               country]['AverageTemperature'].mean())
    max_temp_list.append(global_temp_country[global_temp_country['Country'] ==
                                         country]['AverageTemperature'].max())
    min_temp_list.append(global_temp_country[global_temp_country['Country'] ==
                                         country]['AverageTemperature'].min())
global_temp = global_temp.reset_index()
global_temp['Country'] = countries
global_temp['Max Temp'] = max_temp_list
global_temp['Min Temp'] = min_temp_list
global_temp['Avg Temp'] = mean_temp

#creating df with required columns

global_temp = global_temp[['Country','Max Temp','Min Temp','Avg Temp']]
print(global_temp.head())

#plotting correlation
corr = global_temp.corr()
print(corr)
sns.heatmap(corr)
plt.show()

#training  uning linear reg
X = global_temp['Min Temp'].values.reshape(-1,1)
y = global_temp['Avg Temp'].values.reshape(-1,1)
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


