import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

data_ = []
for l in open('flavors_of_cacao.csv').readlines()[1:]:
    data_.append(l.split(',')[5])
data1= []
for i in open('flavors_of_cacao_2.csv').readlines()[1:]:
    data1.append(i.split(',')[5])
res = []
res = data1 + data_
# mapper
# convert list of countries to list of tuples with '1' near each country
data = list(map(lambda x: (x, 1), res))
# merge-sort
data.sort()

# summarize all numeric fields for each instance

# 6. за один проход по списку
i = 0
d = {}
while i < len(res):
    d[res[i]] = data.count(data[i])
    i += data.count(data[i])
print(d)

# reduce
country = sorted(d.items(), key=lambda item: item[1], reverse=True)
df = pd.DataFrame(country, columns=['name', 'count'])
print(df)

plt.grid()
plt.title("Количество повторений слов")
plt.tick_params(axis='x', rotation=90)
plt.plot(list(df['name']), df['count'], 'm')
plt.show()