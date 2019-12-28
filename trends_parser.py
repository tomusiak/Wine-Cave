from pytrends.dailydata import *
import pandas as pd
import numpy
from sklearn.preprocessing import MinMaxScaler

def processTrends(candidates, startYear, startMonth, endYear, endMonth):
    df_list = []
    for candidate in candidates:
        df = pd.DataFrame(get_daily_data(candidate, startYear, startMonth, endYear, endMonth))
        scaler = MinMaxScaler()
        df[[candidate]] = scaler.fit_transform(df[[candidate]])
        parsed_data = df.drop(columns = [candidate + "_monthly",candidate + "_unscaled","isPartial", "scale"])
        parsed_data.reset_index(level=0, inplace=True) 
        parsed_data["date"] = parsed_data["date"].apply(lambda x: x.strftime('%Y/%m/%d'))        
        parsed_data = convertedDates(parsed_data)
        df_list.append(parsed_data)
    return df_list

def convertedDates(data):
    numbering = -1
    previous_date = ""
    for index, row in data.iterrows():
        if (row['date'] == previous_date):
            data.at[index,'date'] = numbering
        else:
            previous_date = row['date']
            numbering = numbering+1
            data.at[index,'date'] = numbering
    return data