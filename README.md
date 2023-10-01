# Все мои проекты по обработке и визуализации данных

## Excel project 

> Проект выполнен в Microsoft Excel 2019 

### Dashboard + График-карта, выполнены по данным, конвертированным из csv-файла. 

Изначально датасет был взят с сайта [kaggle.com](https://www.kaggle.com/datasets/abhishekpatel/flavors-of-cacaocsv) и поделен на 2 папки (первая - 1000 строк, вторая - 797 строк).

> Полученный csv-файл так же прикреплен в файле. 

Из csv-файла были удалены все строки, содержащие пропуски в данных. И только после этого был создан Dashboard и График-карта:

перед их созданием были составлены промежуточные листы со сводными таблицами по данным, из которых уже берутся обобщенные, отфильтрованные данные на вывод в Dashboard и График-карту. 

> Грaфик-карта по сути является частью Dashbord.

*Dashbord визуализирует:*

- Средний процент содержания какао в плитке по странам, фильтруя данные по годам
- Средняя оценка шоколада по странам, фильтруя их по годам

*График-карта визуализирует:*

- REF стран по годам

## Python projects 

> Библиотеки pandas, numpy, matplotlib, seaborn, plotly, BeautifulSoup, request, functools, dash

**1. Красивая визуализация данных, взятых из [статьи о выбросах углекислого газа по странам](https://en.wikipedia.org/wiki/List_of_countries_by_carbon_dioxide_emissions) с Википедии.**

 > если программа на вашем компьютере не запускается, то скорее всего у вас не установлена библиотека plotly или bs4 или обе библиотеки сразу. 

    pip install plotly-express
    
    pip install bs4
    
> или у вас не установлен менеджер пакетов xlml, тогда:

```
pip install lxml
```
__1й файл:__

**Анимированная диаграмма.**

 Данные взяты с [guthubusercontent](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv).

__2й файл:__

 Парсим страницу [Википедии](https://en.wikipedia.org/wiki/List_of_countries_by_carbon_dioxide_emissions) про выбросы углекислого газа в атмосферу. 

 Считываем необходимые нам строки из html-файла. Создаем датафрейм, в данном случае берем информацию только за 2018 год. Убираем в ручную 2 строки на которых невозожно конвертировать значения из столбца 'Total excluding LUCF'.

 Производим генерацию цветов для диаграмм и наслаждаемся нашей визуализацией.

**Диаграммы**:

 **1.** *выводятся прямо на компьютере*:
 - Круговая(#1: для неотсортированных по значению данных, #2: для отсортированных)
 
 **2.** *выводятся в окно браузера(созданы с помощью plotly.express)(интерактивные)*
 - Древовидная
 - Интерактивная гистограмма
 - Лепестковая(для отсортированных данных)
 - Пузырьковая(1: прямая, 2: круговая)
   > Для них сначала создаем надписи(метки) и параметры х и у для расположения "пузырьков на форме"
 
 **3.** *выводит несколько маленьких, а затем собарет их в одну*
 - Вафельная диаграмма
   > Сначала создаем идет метод для рисования отдельных графиков для каждой представленной в списке страны, а затем в методе get_collage мы собираем их в одну картинку Collage_waffel

**2. Dash projects.**

Проекты, реализованные с помощью библитеки Python dash.
Чтобы установить библиотеку, в командной строке вводим:

```
pip install dash
```

**3. MapReduce для данных по ароматам какао.**

 Использование стека MapReduce для быстрого подсчета количества повторений слов в столбце.

- чтение 6 столбца каждой строки из 2х csv-файлов
- общая сортировка данных
- вывод в консоль map-данных
- конвертация данных типа list в Dataframe (с помощью библиотеки pandas)
- визуализация

**4. Визуализация распределения вероятностей.**

> Реализуется с помощью библиотек matplotlib, numpy, scipy.

![Формулы распределения вероятностей](https://github.com/Khazovalexandra/my_visualization_projects/blob/main/image-40-1024x572.png)

 - Визуализация распределения Бернулли(Bern_distribution). 
 - Биномиальное распределение(Binom_distribution).
 - Геометрическое распределение(Geom_distribution).
 - Распределение Пуассона(Pois_distribution).
 - Отрицательное биномиальное распределение(Negativ_Binom_distribution).
 - Гипергеометрическое распределение(Hyper_Geom_distribution).
 - Дискретное равномерное распределение(Discret_Unif_distribution).
 - Равномерное распределение(Unif_distribution).
 - Нормальное распределение(Normal_distribution).
 - Экспоненциальное распределение(Expo_distribution).
