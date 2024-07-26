import pandas as pd
import seaborn as sns
import jinja2
import numpy as np
import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import matplotlib.pyplot as plt

#reading csv
Air_qty_df = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/cities_air_quality.csv')
Air_qty_df1 = pd.read_csv('C:/Users/DELL/Downloads/Dataset/MyDF/cities_air_quality - Copy.csv')

# drop unnecessary columns
Air_qty_df = Air_qty_df[[' Country', 'AirQuality']]
countries = Air_qty_df[' Country'].unique()


#calculating airquality per country
Air_qty = []
for i in countries:
        Air_qty.append(Air_qty_df[Air_qty_df[' Country'] == i]['AirQuality'].mean())

Air_qty_df1['Country'] = countries
Air_qty_df1['air_qty'] = Air_qty
Air_qty_df1 = Air_qty_df1[['Country','air_qty']]
Air_qty_df1 = Air_qty_df1[Air_qty_df1['air_qty'].notna()]
print(Air_qty_df1.head())

# sorting
Air_qty,countries = (list(x) for x in
                              zip(*sorted(zip(Air_qty,countries), key=lambda pair: pair[0], reverse=True)))

# ploting aqi of all the countries

trace0 = go.Scatter(
    x = countries,
    y = Air_qty,
    fill= None,
    mode='lines',
    name='Uncertainty top',
    line=dict(
        color='rgb(0, 255, 255)',
    )
)


data = [trace0]

layout = go.Layout(
    xaxis=dict(title='Countries'),yaxis=dict(title='AQI'),
    title='AQI of countries around the world',
    showlegend = False)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

#plotting countries with aqi

f, ax = plt.subplots(figsize=(8, 8))
sns.barplot(x=Air_qty[:10], y=countries[:10], palette=sns.color_palette("coolwarm", 25), ax=ax)

texts = ax.set(ylabel="", xlabel="AQI", title="Countries with the highest AQI")
plt.show()

