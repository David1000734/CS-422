import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# ******************** REMOVE ********************
# https://mkang.faculty.unlv.edu/teaching/CS489_689/code3/Linear_Regression.html
# https://www.youtube.com/watch?v=VmbA0pi2cRQ
# https://www.youtube.com/watch?v=P8hT5nDai6A
# ******************** REMOVE ********************
data = pd.read_csv("auto-mpg.data.csv")

# Function will be utilizeing the Min-Max Scaling
def normalize_Data(data_Column):
    """
    normalize_Data will utilize Min-Max Scaling to set
    values between 0 - 1.
    
    :param data_Column: Recieve the column to be normalized
    :return: Output the normalized column in dataType List
    """
    new_Data_List = []

    # Bottom value will remain constent throughout loop
    bottom = max(data_Column) - min(data_Column)        # x_Max - x_Min

    # Traverse each value within data_Column to perform calculation
    for i in range(len(data_Column)):
        # X_norm = (x - x_Min) / (X_Max - X_Min)
        top = data_Column[i] - min(data_Column)         # x - x_Min
        result = top / bottom                   # Top / Bot
        new_Data_List.append(result)
    # For, END
    return new_Data_List
# Normalize_Data funct, END

# To be removed...
def mean_Error(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].mpg
        y = points.iloc[i].horsepower
        total_Error += (y - (m * x + b)) ** 2
    total_Error / float(len(points))
# To be removed...

# ******************** Main ********************

# Drop the last row (Car Name)
data = data.iloc[:, :-1]

for i in range(len(data.columns)):
    #pd.DataFrame(normalize_Data(data.iloc[:, i]))          # To change to DataFrame...
    print(pd.DataFrame(normalize_Data(data.iloc[:, i])))    # Leave as list for now
# For, END

