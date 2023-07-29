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

> Библиотеки pandas, numpy, matplotlib, seaborn, BeautifulSoup, request, functools

**1. MapReduce для данных по ароматам какао.**

 Использование стека MapReduce для быстрого подсчета количества повторений слов в столбце.

- чтение 6 столбца каждой строки из 2х csv-файлов
- общая сортировка данных
- вывод в консоль map-данных
- конвертация данных типа list в Dataframe (с помощью библиотеки pandas)
- визуализация
