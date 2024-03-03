import pandas as pd
import numpy as np
import math

# ******************** REMOVE ********************
# https://mkang.faculty.unlv.edu/teaching/CS489_689/code3/Linear_Regression.html
# https://www.youtube.com/watch?v=VmbA0pi2cRQ
# ******************** REMOVE ********************



data = pd.read_csv("auto-mpg.data.csv")

# Drop the last row (Car Name)
data = data.iloc[:, :-1]
print(data)


