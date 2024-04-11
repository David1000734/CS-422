import pandas as pd
import numpy as np


# ******************** Main ********************
# Read data in and specify header at 0th index
data = pd.read_csv("MNIST.csv", header = 0)

# Get y values
y_actual = data.label

# Get x values
x_values = data.iloc[ :, 1:]

print(data)

print(y_actual)
print(x_values)
