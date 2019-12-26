print("Horse")
from pytrends.dailydata import *
import pandas as pd
import numpy

df = pd.DataFrame(get_daily_data("Joe Biden", 2019, 10, 2019,12))
export_csv = df.to_csv(r'biden.csv', index = None, header=True)
print(df.head())
df.to_csv(index=False)
input(".")