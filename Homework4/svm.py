import pandas as pd
import numpy as np


# ******************** Main ********************
# Reading data from csv, specifying header is at the 0'th position
data = pd.read_csv("MNIST_HW4.csv", header = [0])

# Pull the label with all the numbers out
y_actual = data.label
# Take only the values from the array out
x_values = data.iloc[ :, 1 :]
# Pick X train and test values later in the code.

print(y_actual)     # DEBUG
print(x_values)     # DEBUG

# Original file size is 1000. Remove header so size is 999
