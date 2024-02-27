import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('MNIST_100.csv')

y = data.iloc[:, 0]         # Only take first column
X = data.drop('label', axis=1)      # Drop first row
print(X.shape)
print(y.shape)

# ******************** Task 1 ********************
# Reduced to 2 components
pca = PCA(n_components=2)
pca.fit(X)
PCAX = pca.transform(X)

plt.title("MNIST Data")
plt.xlabel("X-Pixel")
plt.ylabel("Y-Pixel")
plt.plot(PCAX[:, 0], PCAX[:, 1], 'wo', )
for i in range(len(y)):
    # Plots values from i to i+1 for both the x and y axis
    # using the value from y[i]
    plt.text(PCAX[i:i+1, 0], PCAX[i:i+1, 1], y[i])
plt.show()

plt.figure(figsize=(10, 8))     # larger Graph
plt.title("MNIST Data")
plt.xlabel("X-Pixel")
plt.ylabel("Y-Pixel")
# Prints number from 0 to 10
for i in range(10):
    # Plot Graph and lable each the cooresponding number
    plt.scatter(PCAX[i * 100:(i + 1) * 100, 0], PCAX[i * 100:(i + 1) * 100, 1], label=i)
# Create legend for all values
plt.legend()
plt.show()

# ******************** Task 1 ********************

# ******************** Task 2 ********************
house_Data = pd.read_csv("housing_training.csv")

# Drop all rows except K, L, M, N
hData = house_Data.iloc[: , 10:14]

# Drop row L
hData.drop(columns=hData.columns[1], axis=1, inplace=True)

plt.figure(figsize= (15, 8))     # increase image size for text overlap
plt.title("Distributions")      # title
plt.ylabel("Quantity")
plt.grid()                      # Style, adding grid to plot

# Labels for columns
plt.boxplot(hData,
            labels= ['Pupil-teacher ratio',
                '% lower status of the population',
                'Median value of owner-occupied homes in $1000s'])
plt.show()

# ******************** Task 2 ********************

# ******************** Task 3 ********************
# since house_Data still has the full csv of the file
house_Data = house_Data.iloc[:,: 1]     #A Take only the first column

plt.title("Per Capita Crime Rate")
plt.ylabel("Quantity")
plt.xlabel("Crime Rate")

plt.hist(house_Data)
plt.show()

# ******************** Task 3 ********************


