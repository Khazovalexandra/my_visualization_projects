import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as ani

#Анимированная диаграмма

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
df = pd.read_csv(url, delimiter=',', header='infer')
df_interest = df.loc[
            	df['Country/Region'].isin(['United Kingdom', 'US', 'Italy', 'Germany'])
            	& df['Province/State'].isna()]
df_interest.rename(
            	index=lambda x: df_interest.at[x, 'Country/Region'], inplace=True)
df1 = df_interest.transpose()
df1 = df1.drop(['Province/State', 'Country/Region', 'Lat', 'Long'])
df1 = df1.loc[(df1 != 0).any(1)]
df1.index = pd.to_datetime(df1.index)

color = ['red', 'green', 'blue', 'orange']
fig = plt.figure()
plt.xticks(rotation=45, ha="right", rotation_mode="anchor") #циклическое перемещение значений по оси x
plt.subplots_adjust(bottom = 0.2, top = 0.9) #убеждаемся, что даты (по оси x) умещаются на экране
plt.ylabel('No of Deaths')
plt.xlabel('Dates')

def buildmebarchart(i=int):
    plt.legend(df1.columns)
    p = plt.plot(df1[:i].index, df1[:i].values) #обращаем внимание, что происходит возврат набора данных вплоть до точки i
    for i in range(0,4):
        p[i].set_color(color[i]) #устанавливаем цвет для каждой кривой
		
animator = ani.FuncAnimation(fig, buildmebarchart, interval = 100)
plt.show()