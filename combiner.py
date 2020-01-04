from pytrends.dailydata import *
import pandas as pd
import numpy
import copy

def combineData(predictit_data, trends_data, candidates):
    combined_data = predictit_data
    row_count = len(predictit_data.index)
    trends_list = []
    initialized_rows = [0] * row_count
    combined_data.insert(1, "Trends", initialized_rows, True)
    for index, row in combined_data.iterrows():
        for trends_data_candidate in trends_data:
            for index2, row2 in trends_data_candidate.iterrows():
                candidate_name_in_trends = row2.index[1]
                if (candidate_name_in_trends in candidates):
                    if (row[candidate_name_in_trends]==1):
                        predictDate = row['Date']
                        trendDate = row2['date']
                        if (predictDate == trendDate):
                                trends_list.append(row2[candidate_name_in_trends])
    combined_data["Trends"] = trends_list
    num_delete = len(candidates)*5*-1
    combined_data = combined_data[:num_delete]
    return combined_data