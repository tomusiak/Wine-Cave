import pandas as pd
import numpy

def processData(raw_data, candidates):
    raw_data = pd.read_csv(raw_data)
    parsed_data = raw_data.drop(columns = ["OpenSharePrice","HighSharePrice","LowSharePrice", "TradeVolume"])
    parsed_data = removeIrrelevants(parsed_data, candidates)
    parsed_data = convertDates(parsed_data)
    final_table = restructure(parsed_data, candidates)
    return final_table
    
def removeIrrelevants(data, candidates):
    for index, row in data.iterrows():
        if (row['ContractName'] not in candidates):
            data.drop(index,inplace = True)
    return data
    
def convertDates(data):
    numbering = -1
    previousDate = ""
    for index, row in data.iterrows():
        if (row['Date'] is previousDate):
            data.at[index,'Date'] = numbering
        else:
            previousDate = row['Date']
            numbering = numbering+1
            data.at[index,'Date'] = numbering
    return data
    
def restructure(data, candidates):
    column_names = candidates
    column_names.insert(0,'Date')
    column_names.append('Price')
    zeros = [0] * len(column_names)
    empty_data = dict(zip(column_names,zeros))
    single_row = pd.DataFrame([empty_data])
    full_table = pd.DataFrame(columns=empty_data)
    for index, row in data.iterrows():
        date = row['Date']
        name = row['ContractName']
        price = row['CloseSharePrice']
        single_row['Date'] = date
        single_row[name] = 1
        single_row['Price'] = price
        full_table = full_table.append(single_row)
        single_row = pd.DataFrame([empty_data])
    return full_table