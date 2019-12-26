from pytrends.dailydata import *
import pandas as pd
import numpy

def definitely_a_function():
    df = pd.DataFrame(get_daily_data("Joe Biden", 2019, 10, 2019,12))
    export_csv = df.to_csv(r'Biden.csv', index = None, header=True)
    print(df.head())
    df.to_csv(index=False)
    input(".")