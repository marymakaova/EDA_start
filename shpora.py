import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sps

# 1. Загрузка данных
df = pd.read_csv('data.csv')

# 2. Первичный осмотр
print(df.head())
print(df.info())
print(df.describe())

# 3. Пропуски и дубликаты
print(df.isnull().sum())
print(df.duplicated().sum())

# 4. Распределения числовых переменных
df.hist(figsize=(12, 8), bins=30)
plt.tight_layout()

# 5. Категориальные переменные
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head())

# 6. Корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)

# 7. Выбросы (боксплоты)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(df.select_dtypes(include=[np.number]).columns[:3]):
    sns.boxplot(y=df[col], ax=axes[i])

# 8. Тесты на нормальность для ключевых переменных
for col in ['important_numeric_col']:
    stat, p = sps.shapiro(df[col].dropna())
    print(f"{col}: Shapiro-Wilk p-value = {p:.4f}")

# 9. numpy

#база
np.array(data)                # Создание массива
np.arange(start, stop, step)  # Массив с равномерным шагом
np.linspace(start, stop, num) # Массив с заданным числом точек
np.random.randn(n)            # Генерация нормальных случайных чисел

#статистика
np.mean(arr)                  # Среднее значение
np.median(arr)                # Медиана
np.std(arr, ddof=1)           # Стандартное отклонение (ddof=1 для несмещенной)
np.var(arr, ddof=1)           # Дисперсия
np.percentile(arr, [25, 75])  # Квантили
np.min(arr), np.max(arr)      # Минимум и максимум
np.quantile(arr, 0.5)         # Квантиль (0.5 = медиана)

#форма данных
arr.shape                     # Размерность массива
arr.ndim                      # Количество измерений
arr.size                      # Общее количество элементов
arr.dtype                     # Тип данных

# 10. Pandas

#чтение данных
pd.read_csv('file.csv')       # Чтение CSV
pd.read_excel('file.xlsx')    # Чтение Excel
df.to_csv('output.csv')       # Сохранение в CSV

#просмотр данных
df.head(n=5)                  # Первые n строк
df.tail(n=5)                  # Последние n строк
df.info()                     # Информация о датафрейме
df.describe()                 # Статистика по числовым колонкам
df.describe(include='all')    # Статистика по всем колонкам
df.dtypes                     # Типы данных колонок
df.columns                    # Названия колонок
df.index                      # Индексы

#базовые операции
df['column']                  # Выбор колонки
df[['col1', 'col2']]          # Выбор нескольких колонок
df.iloc[row_index]            # Выбор по индексу
df.loc[row_label]             # Выбор по метке
df.query('condition')         # Фильтрация через запрос

#статистика
df.mean()                     # Среднее по колонкам
df.median()                   # Медиана
df.std()                      # Стандартное отклонение
df.var()                      # Дисперсия
df.skew()                     # Асимметрия
df.kurt()                     # Эксцесс
df.corr()                     # Матрица корреляций
df.corrwith(df['col'])        # Корреляция с конкретной колонкой

#группировка и агрегация
df.groupby('column').mean()   # Группировка и агрегация
df.groupby(['col1', 'col2']).agg(['mean', 'std'])
df.pivot_table(values='val', index='idx', columns='col', aggfunc='mean')
pd.crosstab(df['A'], df['B']) # Таблица сопряженности

#дубликаты/пропуски
df.isnull().sum()             # Количество пропусков по колонкам
df.dropna()                   # Удаление строк с NaN
df.fillna(value)              # Заполнение пропусков
df.duplicated().sum()         # Количество дубликатов
df.drop_duplicates()          # Удаление дубликатов

#изменение данных
df.rename(columns={'old': 'new'})  # Переименование колонок
df.sort_values('column')           # Сортировка
df.value_counts()                  # Подсчет уникальных значений
df.nunique()                       # Количество уникальных значений
df.memory_usage()                  # Использование памяти

# 11. Matplotlib

#создание графиков
plt.figure(figsize=(w, h))         # Создание фигуры
plt.subplots(nrows, ncols)         # Сетка графиков
fig, axes = plt.subplots(2, 2)     # Создание осей
plt.subplot(2, 2, 1)               # Выбор конкретной позиции

#типы графиков
plt.plot(x, y)                     # Линейный график
plt.scatter(x, y)                  # Точечная диаграмма
plt.hist(data, bins=30)            # Гистограмма
plt.boxplot(data)                  # Боксплот
plt.bar(categories, values)        # Столбчатая диаграмма
plt.pie(sizes, labels=labels)      # Круговая диаграмма

#настройка
plt.title('Title')                 # Заголовок
plt.xlabel('X label')              # Подпись оси X
plt.ylabel('Y label')              # Подпись оси Y
plt.legend()                       # Легенда
plt.grid(True)                     # Сетка
plt.tight_layout()                 # Автоподгонка
plt.xlim(min, max)                 # Лимиты оси X
plt.ylim(min, max)                 # Лимиты оси Y
plt.xticks(rotation=45)            # Поворот подписей
plt.savefig('plot.png', dpi=300)   # Сохранение
plt.show()                         # Отображение

#работа с осями
ax.set_title()                     # Установка заголовка для оси
ax.set_xlabel()                    # Подпись оси X для оси
ax.set_ylabel()                    # Подпись оси Y для оси
ax.tick_params()                   # Настройка тиков

# 12. Seaborn

#распределения
sns.histplot(data, kde=True)       # Гистограмма с KDE
sns.kdeplot(data)                  # Оценка плотности
sns.rugplot(data)                  # Коврик (rug plot)
sns.boxplot(x='cat', y='num', data=df)  # Боксплот
sns.violinplot(x='cat', y='num', data=df) # Скрипичная диаграмма

#категориальные данные
sns.countplot(x='category', data=df)     # Подсчет категорий
sns.barplot(x='cat', y='num', data=df)   # Столбчатая с доверительными интервалами
sns.catplot(kind='box', x='cat', y='num', data=df) # Фасетный категориальный

#взаимосвязи
sns.scatterplot(x='col1', y='col2', data=df, hue='cat') # Точечная
sns.lineplot(x='col1', y='col2', data=df)               # Линейная
sns.regplot(x='col1', y='col2', data=df)                # Регрессия с доверительным интервалом
sns.lmplot(x='col1', y='col2', data=df, hue='cat')      # Регрессия по категориям

#матрицы и корреляции
sns.heatmap(corr_matrix, annot=True)      # Тепловая карта
sns.clustermap(corr_matrix)               # Кластеризованная тепловая карта
sns.pairplot(df, hue='category')          # Парные диаграммы рассеяния
sns.jointplot(x='col1', y='col2', data=df, kind='scatter') # Совместное распределение

#многомерные
sns.FacetGrid(df, col='cat_col')          # Сетка графиков по категориям
sns.PairGrid(df, vars=['col1', 'col2'])   # Полная кастомизация pairplot

#стили и цвета
sns.set_style('whitegrid')                # Стиль оформления
sns.set_palette('husl')                   # Цветовая палитра
sns.color_palette('coolwarm', as_cmap=True) # Цветовая карта
sns.despine()                             # Убрать верхнюю и правую оси

#scipy.stats

#описательная статистика
sps.describe(data)                # Полное описание (n, minmax, mean, var, skew, kurt)
sps.mode(data)                    # Мода (самое частое значение)
sps.iqr(data)                     # Межквартильный размах
sps.zscore(data)                  # Z-оценки (стандартизация)
sps.skew(data)                    # Коэффициент асимметрии
sps.kurtosis(data)                # Коэффициент эксцесса
sps.moment(data, moment=2)        # Центральный момент 2-го порядка = дисперсия

#тесты на нормальность
sps.shapiro(data)                 # Тест Шапиро-Уилка (n < 5000)
sps.normaltest(data)              # Тест на нормальность Д'Агостино-Пирсона
sps.kstest(data, 'norm')          # Тест Колмогорова-Смирнова

#параметрические тесты
sps.ttest_1samp(data, popmean)    # Одновыборочный t-тест
sps.ttest_ind(sample1, sample2)   # Двухвыборочный t-тест
sps.ttest_rel(sample1, sample2)   # T-тест для парных выборок
sps.pearsonr(x, y)                # Корреляция Пирсона

#непараметрические тесты
sps.mannwhitneyu(sample1, sample2) # U-тест Манна-Уитни
sps.wilcoxon(sample1, sample2)     # Тест Уилкоксона для парных выборок
sps.kruskal(*samples)              # Критерий Краскела-Уоллиса (3+ групп)
sps.spearmanr(x, y)                # Корреляция Спирмена
sps.kendalltau(x, y)               # Корреляция Кендалла

#распределения
sps.norm.ppf(0.975)               # Квантиль нормального распределения
sps.t.ppf(0.975, df=10)           # Квантиль t-распределения
sps.chi2.ppf(0.95, df=5)          # Квантиль хи-квадрат
sps.f.ppf(0.95, dfn=3, dfd=10)    # Квантиль F-распределения
sps.uniform(loc=a, scale=b-a)     # Равномерное распределение

#сравнение распределений
sps.ks_2samp(sample1, sample2)    # Тест Колмогорова-Смирнова для 2 выборок
sps.anderson_ksamp([sample1, sample2]) # Тест Андерсона-Дарлинга для k выборок
sps.chi2_contingency(contingency_table) # Тест хи-квадрат для таблиц сопряженности



