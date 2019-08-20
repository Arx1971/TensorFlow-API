from __future__ import print_function
import pandas as pd

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

california_housing_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.head()

print(california_housing_dataframe)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

data = pd.DataFrame({ 'City name': city_names, 'Population': population })
california_housing_dataframe.hist('housing_median_age')
print(data)