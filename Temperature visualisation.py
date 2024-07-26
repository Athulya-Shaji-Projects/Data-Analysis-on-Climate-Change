import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import time

#reading csv
global_temp = pd.read_csv('C:/Users/DELL/Downloads/TempChange/GlobalTemperatures.csv')

#extracting year from date
years = np.unique(global_temp['dt'].apply(lambda x: x[:4]))
#calculating mean temp per year
mean_temp = []
for year in years:
    mean_temp.append(global_temp[global_temp['dt'].apply(
        lambda x: x[:4]) == year]['LandAverageTemperature'].mean())

#reading csv
temp_by_country = pd.read_csv('C:/Users/DELL/Downloads/TempChange/GlobalLandTemperaturesByCountry.csv')
countries = temp_by_country['Country'].unique()

#calculating min and max temp differance for each country
max_min = []

for country in countries:
    curr_temps = temp_by_country[temp_by_country['Country'] == country]['AverageTemperature']
    max_min.append((curr_temps.max(), curr_temps.min()))
max_min_1 = []
countries1 = []

for i in range(len(max_min)):
    if not np.isnan(max_min[i][0]):
        max_min_1.append(max_min[i])
        countries1.append(countries[i])

diff = []
for tpl in max_min_1:
    diff.append(tpl[0] - tpl[1])

diff, countries1 = (list(x) for x in
                              zip(*sorted(zip(diff, countries1), key=lambda pair: pair[0], reverse=True)))

# ploting avg temp over the years

trace0 = go.Scatter(
    x = years,
    y = mean_temp,
    name='Average Temperature',
    line=dict(
        color='rgb(199, 121, 093)',
    )
)
data = [trace0]

layout = go.Layout(
    xaxis=dict(title='Years'),yaxis=dict(title='Average Temperature'),
    title='Average land temperature of Earth from 1750 to 2015',
    showlegend = False)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

#plotting countries with max temp differance

f, ax = plt.subplots(figsize=(8, 8))
sns.barplot(x=diff[:10], y=countries1[:10], palette=sns.color_palette("coolwarm", 25), ax=ax)

texts = ax.set(ylabel="", xlabel="Temperature difference", title="Countries with the highest temperature differences")
plt.show()


