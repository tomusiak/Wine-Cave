import pandas as pd
import numpy
import copy

def processData(raw_data, candidates):
    raw_data = pd.read_csv(raw_data)
    parsed_data = raw_data.drop(columns = ["OpenSharePrice","HighSharePrice","LowSharePrice", "TradeVolume"])
    parsed_data = removeIrrelevants(parsed_data, candidates)
    parsed_data = convertDates(parsed_data)
    restructured_table = restructure(parsed_data, candidates)
    final_table = createDifferential(restructured_table, candidates)
    return final_table
    
def removeIrrelevants(data, candidates):
    for index, row in data.iterrows():
        if (row['ContractName'] not in candidates):
            data.drop(index,inplace = True)
    return data
    
def convertDates(data):
    numbering = -1
    previous_date = ""
    for index, row in data.iterrows():
        if (row['Date'] == previous_date):
            data.at[index,'Date'] = numbering
        else:
            previous_date = row['Date']
            numbering = numbering+1
            data.at[index,'Date'] = numbering
    return data
    
def restructure(data, candidates):
    column_names = copy.deepcopy(candidates)
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
    
def createDifferential(data, candidates):
    push_length = len(candidates)*5
    row_count = len(data.index)
    initialized_rows = [0] * row_count
    data.insert(len(data.columns), "Predictions Diff", initialized_rows, True)
    new_data = data.shift(periods=push_length)
    current_prices = data.loc[:,['Price']]
    late_prices = new_data.loc[:,['Price']]
    current_prices['Price'] = current_prices['Price'].str.replace('$', '')
    late_prices['Price'] = late_prices['Price'].str.replace('$', '')
    diff = (current_prices.astype(float) - late_prices.astype(float))/current_prices.astype(float)
    data["Predictions Diff"] = diff
    return data
            
            