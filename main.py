import pandas as pd
import numpy
from predict_parser import *
from trends_parser import *

#Set these values only
candidates = ["Biden","Warren","Sanders","Buttigieg","Klobuchar","Yang","Booker","Steyer","Bloomberg"]
predictit_data = "PredictItData.csv"
data = processData(predictit_data, candidates)
#export_csv = data.to_csv(r'processedPredictIt.csv', index = None, header=True)
## Export to save time if desired
print(data)
input(".")