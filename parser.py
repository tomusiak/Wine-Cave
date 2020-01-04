import pandas as pd
import numpy
from predict_parser import *
from trends_parser import *
from combiner import *

#Set these values only
directory = "data/"
candidates = ["Joe Biden","Elizabeth Warren","Bernie Sanders","Pete Buttigieg","Amy Klobuchar","Andrew Yang","Cory Booker","Tom Steyer","Michael Bloomberg"]
startYear = 2019
startMonth = 10
endYear = 2019
endMonth = 12
predictit_file = "PredictItData.csv"
predictit_data = processData(directory + predictit_file, candidates)
trends_data = processTrends(candidates, startYear, startMonth, endYear, endMonth)
combined_data = combineData(predictit_data, trends_data, candidates)
export_csv = combined_data.to_csv(r'data\FinalCombined.csv', index = None, header=True)
print(combined_data)