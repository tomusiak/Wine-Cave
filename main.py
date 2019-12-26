import pandas as pd
import numpy
from predict_parser import *
from trends_parser import *

#Set these values only
directory = "data/"
candidates = ["Biden","Warren","Sanders","Buttigieg","Klobuchar","Yang","Booker","Steyer","Bloomberg"]
startYear = 2019
startMonth = 10
endYear = 2019
endMonth = 12
predictit_file = "PredictItData.csv"
predictit_data = processData(directory + predictit_file, candidates)
#export_csv = data.to_csv(r'processedPredictIt.csv', index = None, header=True)
## Export to save time if desired
trends_data = processTrends(candidates, startYear, startMonth, endYear, endMonth)
print(trends_data)
input(".")