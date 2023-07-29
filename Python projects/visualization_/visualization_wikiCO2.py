import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import matplotlib.animation as ani
import numpy as np
from pywaffle import Waffle
from PIL import Image
import math

#парсинг таблицы с Википедии 
#статья по выбросам углекислого газа в странах
#https://en.wikipedia.org/wiki/List_of_countries_by_carbon_dioxide_emissions

wikiurl='https://en.wikipedia.org/wiki/List_of_countries_by_carbon_dioxide_emissions'
table_class='wikitable sortable jquery-tablesorter'
response=requests.get(wikiurl)
#status 200: Сервер успешно ответил на запрос http
print(response.status_code)

soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table',{'class':"wikitable"})
df2018 = pd.read_html(str(table))[0]
print(df2018)

# Получаем списки данных
emi_ = df2018[('2018 CO2 emissions[20]', 'Total excluding LUCF[22]')]
country_ = list(df2018[('Country[19]', 'Country[19]')])
country_mod = [i.replace('\xa0',' ') for i in country_]

# Создаем DataFrame
df = pd.DataFrame(zip(country_mod,emi_), columns = ['countries', 'emission_2018'])

# Убираем строку о стране, которую нельзя конвертировать
df = df[df['countries']!='Serbia & Montenegro']  
df = df[df['countries']!='Japan']  
df.iloc[:,1] = df.iloc[:,1].astype('float')
df = df[(df['emission_2018']>200) & (df['emission_2018']<1000)]
df['percentage'] = [i*100/sum(df['emission_2018']) for i in df['emission_2018']]
df.head(9)

df_s = df.sort_values(by='emission_2018', ascending=False)
df_s.head(9)

#Генератор красивых цветов для графиков
def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(df))
pal_plas = get_color('plasma_r', len(df))
pal_spec = get_color('Spectral', len(df))
pal_hsv = get_color('hsv', len(df))

#Круговая диаграмма

plt.gcf().set_size_inches(12, 12)
sns.set_style('darkgrid')

# Установим максимальное значение
max_val = max(df['emission_2018'])*1.01
ax = plt.subplot(projection='polar')

# Зададим внутренний график
ax.set_theta_zero_location('N')
ax.set_theta_direction(1)
ax.set_rlabel_position(0)
ax.set_thetagrids([], labels=[])
ax.set_rgrids(range(len(df)), labels= df['countries'])

# Установим проекцию
ax = plt.subplot(projection='polar')

for i in range(len(df)):
    ax.barh(i, list(df['emission_2018'])[i]*2*np.pi/max_val,
            label=list(df['countries'])[i], color=pal_vi[i])

plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Для отсортированных данных

plt.gcf().set_size_inches(12, 12)
sns.set_style('darkgrid')

# Установим максимальное значение
max_val = max(df_s['emission_2018'])*1.01
ax = plt.subplot(projection='polar')

for i in range(len(df)):
    ax.barh(i, list(df_s['emission_2018'])[i]*2*np.pi/max_val,
            label=list(df_s['countries'])[i], color=pal_plas[i])

# Зададим внутренний график
ax.set_theta_zero_location('N')
ax.set_theta_direction(1)
ax.set_rlabel_position(0)
ax.set_thetagrids([], labels=[])
ax.set_rgrids(range(len(df)), labels= df_s['countries'])

# Установим проекцию
ax = plt.subplot(projection='polar')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

#Древовидная диаграмма

fig = px.treemap(df, path=[px.Constant('Countries'), 'countries'],
              values=df['emission_2018'],
              color=df['emission_2018'],
                 color_continuous_scale='Spectral_r',
                 color_continuous_midpoint=np.average(df['emission_2018'])
              )
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

#Интерактивная гистограмма

fig = px.bar(df, x='countries', y='emission_2018', text='emission_2018',
           	color ='countries', color_discrete_sequence=pal_vi)
fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                          	'paper_bgcolor': 'white'})
fig.update_layout(width=1100, height=500,
               margin = dict(t=15, l=15, r=15, b=15))
fig.show()

#Лепестковая диаграмма

#Визуализация отсортированного массива
fig = px.line_polar(df_s, r='emission_2018',
                          	theta='countries', line_close=True)
fig.update_traces(fill='toself', line = dict(color=pal_spec[-5]))
fig.show()

#Пузырьковая диаграмма

#Сначала строим список значений x и y и меток
df_s['X'] = [1]*len(df_s)
list_y = list(range(0,len(df_s)))
list_y.reverse()
df_s['Y'] = list_y
# Столбец меток
df_s['labels'] = ['<b>'+i+'<br>'+format(j, ",") for i,j in zip(df_s['countries'], df_s['emission_2018'])]

#1
fig = px.scatter(df_s, x='X', y='Y',
              color='countries', color_discrete_sequence=pal_vi,
              size='emission_2018', text='labels', size_max=30)
fig.update_layout(width=500, height=1100,
               margin = dict(t=0, l=0, r=0, b=0),
               showlegend=False
              )
fig.update_traces(textposition='middle right')
fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
fig.update_layout({'plot_bgcolor': 'white',
                          	'paper_bgcolor': 'white'})
fig.show()

#2
# Создаем координаты X и Y по кругу
e = 360/len(df)
degree = [i*e for i in list(range(len(df)))]
df_s['X_coor'] = [math.cos(i*math.pi/180) for i in degree]
df_s['Y_coor'] = [math.sin(i*math.pi/180) for i in degree]

fig = px.scatter(df_s, x='X_coor', y='Y_coor',
                 color="countries", color_discrete_sequence=pal_vi,
                 size='emission_2018', text='labels', size_max=40)
fig.update_layout(width=800, height=800,
                  margin = dict(t=0, l=0, r=0, b=0),
                  showlegend=False
                 )
fig.update_traces(textposition='bottom center')
fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.show()

#Вафельная диаграмма

save_name = []
for i,p,n,c in zip(df_s['emission_2018'], df_s['percentage'], df_s['countries'], pal_hsv):
    fig = plt.figure(FigureClass=Waffle,
                     rows=10, columns=20,
                     values=[i, sum(df_s['emission_2018'])-i], 
                     colors=[c,'gainsboro'],
                     labels=[n + ' ' + str(round(p,1)) +' %','Other countries'],
                     figsize = (8,8),
                     legend={'loc':'upper right', 'bbox_to_anchor': (1, 1), 'fontsize':24}
                    )
    save_name.append('Python projects/visualization_/waffle_diagrams/waffle_'+ n + '.png')
    plt.tight_layout()
    plt.savefig('Python projects/visualization_/waffle_diagrams/waffle_'+ n + '.png', bbox_inches='tight')   #export_fig
    #plt.show()

def get_collage(cols_n, rows_n, width, height, input_sname, save_name):
    c_width = width//cols_n
    c_height = height//rows_n
    size = c_width, c_height
    new_im = Image.new('RGB', (width, height))
    ims = []
    for p in input_sname:
        im = Image.open(p)
        im.thumbnail(size)
        ims.append(im)
    i, x, y = 0,0,0
    
    for col in range(cols_n):
        for row in range(rows_n):
            print(i, x, y)
            try:
                new_im.paste(ims[i], (x, y))
                i += 1
                y += c_height
            except IndexError:
                pass
        x += c_width
        y = 0
    new_im.save(save_name)

    width = cols_n * width
    height = rows_n * height

get_collage(5, 5, 2840, 1445, save_name, 'Python projects/visualization_/waffle_diagrams/Collage_waffle.png')