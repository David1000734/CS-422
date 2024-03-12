import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math

# ******************** REMOVE ********************
# https://mkang.faculty.unlv.edu/teaching/CS489_689/code3/Linear_Regression.html
# https://www.youtube.com/watch?v=VmbA0pi2cRQ
# https://www.youtube.com/watch?v=P8hT5nDai6A
# https://www.youtube.com/watch?v=ltXSoduiVwY
# https://www.youtube.com/watch?v=sRh6w-tdtp0
# Multiple linear Regression
# x1, x2, x3, xn
# m1, m2, m3, mn
# Y = m1x1 + m2x2 + m3x3 + mnxn + c
# c -> y-intercept
# ******************** REMOVE ********************
data = pd.read_csv("auto-mpg.data.csv")

# Storing the global min and max of the mpg for later
mpg_Max = 0
mpg_Min = 0

# Function will be utilizeing the Min-Max Scaling
def normalize_Data(data_Column):
    """
    normalize_Data will utilize Min-Max Scaling to set
    values between 0 - 1.
    
    :param data_Column: Recieve the column to be normalized
    :return: Output the normalized column in dataType List
    """
    new_Data_List = []
    # Will be updating the global value so def here
    global mpg_Max
    global mpg_Min

    # Quick check to find the min and max from mpg
    if (data_Column.name == "mpg"):
        mpg_Max = max(data_Column)
        mpg_Min = min(data_Column)

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

def un_normalize_Data(data, max, min):
    """
    Given that we know what the specific data's min and max is, 
    we can get it's original value by the following formula:
    X: is norm-val
    Un-normalized val = X * (max - min) + min

    :param data: Normalized data point
    :param max: Original Max value of data
    :param min: Original Min value of data
    :return: un-normalized data point
    """

    return data * (max - min) + min

# *************** CHANGE THE CODE DOWN HERE ***************
def linear_Model(x_val, y_target, learning_rate, iteration, arr):
    m = y_target.size
    # Create column of 0's of the same size as x's features
    theta = np.zeros((x_val.shape[1], 1))

    for i in range(iteration):
        y_pred = np.dot(x_val, theta)

        # Cost Function
        # 1/2m Sum(y_pred - Y)^2, where Y is the actual value
        cost = ((1 / (2 * m)) * np.sum((y_pred - y_target) ** 2, axis = 0))
        # cost = (1/(2*m))*np.sum(np.square(y_pred - y_target))

        # Gradient Descent
        # d_theta = 1/m(matrix_mul(X^T, y_pred - Y))
        d_theta = ((1 / m) * np.dot(x_val.transpose(), y_pred - y_target))
        # d_theta = (1/m)*np.dot(x_val.T, y_pred - y_target)

        # theta = theta - alpha * d_theta
        theta -= learning_rate * d_theta
        # print(theta)

        # Un-Norm data and store into array
        arr.append(un_normalize_Data(cost, mpg_Max, mpg_Min))

    # Un-Norm data
    for j in range(len(y_pred)):
        # Un-normalize and also round to nearest int
        y_pred[j] = np.round(un_normalize_Data(y_pred[j], mpg_Max, mpg_Min))
        # y_pred[j] = un_normalize_Data(y_pred[j], mpg_Max, mpg_Min)

    return y_pred
# *************** CHANGE THE CODE UP HERE ***************

def k_fold_Test(arr, folds):
    isDivisible = True
    train_arr = pd.DataFrame()        # Val to train the prediction
    test_arr = pd.DataFrame()         # Predicted values from training
    multiplier = 0                    # Used to find what rows to get

    if (folds < 2 or folds > 100):
        print("Error: K-Folds must be higher than 2 and less than 100.")
        return
    # If, END

    # To Ensure we are not geting from test, we must
    # do the getting inside the for loop
    if (len(arr) % folds != 0):
        isDivisible = False
        multiplier = len(arr) / folds
    else :
        multiplier = len(arr) / folds
        multiplier = math.ceil(multiplier)
    # if else, END
    multiplier = int(multiplier)

    # Loop number of folds
    for fold_I in range(folds):
        temp_I = fold_I             # Used for k-fold test
        # Separate data
        # Get rows of data via fold_I and multiplier. use .iloc



        # Find if the folds is divisible by the array.
        # If it is, great, if not, fill in missing by randomly
        # getting columns from the array. Ensure not to grab from test
        if (not isDivisible):
            # Since mod is not 0, fill in missing rows
            rem = len(arr) % folds

            # Iterate however many times we need to fill in rows
            for rand_I in range(rem, (folds - 1)):
                while (temp_I == fold_I):
                    # Rand range between 0 and 10, NOT including 10
                    temp_I = rand.randrange(0, 10)
                    # Keep doing random numbers until one not inside
                    # the test set is picked.
                # While, END

                # Find random val within range
                randRow = rand.randrange(temp_I * multiplier,
                                         (temp_I + 1) * multiplier)
                # Get the row
                dup_Row = arr.iloc[randRow , : ]
                print(dup_Row)                
                test_arr = pd.concat([test_arr, dup_Row], ignore_index = True)
                test_arr.reset_index()
            # For, END
        # If notDivisible, END
        print(test_arr)

        # Loop the entire test set
        for test_I in range(len(train_arr)):
            
            
            
            pass


        # Check predicted value with test array
        for pred_I in range(len(test_arr)):          # Only check however many is in my test array

            pass

        # Clear both arrays
        test_arr = test_arr.iloc[0 : 0]
        train_arr = train_arr.iloc[0 : 0]
    pass
    # For k-folds, END

# Funct K_Fold_Test, END 

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

lr_lambda = 0.5               # Learning Rate
max_Iteration = 10000         # Iterations

# Drop the last row (Car Name)
data = data.iloc[:, :-1]

build_DF = pd.DataFrame()

# Normalize data and rebuild back into dataFrame
for i in range(len(data.columns)):
    # To change to DataFrame...
    temp_DF = pd.DataFrame(normalize_Data(data.iloc[:, i]))
    # Get name of each column
    columnName = "STD_" + data.columns[i]

    # Rebuild the data back into a DataFrame
    build_DF.insert(i, columnName, temp_DF, True)
# For, END

# Normalize all data and then un-normalize later
# X is all values besides MPG
x = build_DF.iloc[ : , 1 : 7]

# Y is the MPG, the predicting value
y = build_DF.iloc[ : , 0 : 1]

# Adding a column of 1's
x["theta_0"] = 1
cost_arr = []

k_fold_Test(build_DF, 10)






# predict = linear_Model(x, y, lr_lambda, max_Iteration, cost_arr)
# print(predict)
# # print(y)
# # print(cost_arr)
# # print(cv)

# if (predict[-1] == 30):
#     print("It's fine")
# else:
#     print("Is not find")

# # for i in range (len(theta)):
# #     print(un_normalize_Data(theta[i], mpg_Max, mpg_Min))

# rng = np.arange(0, max_Iteration)
# plt.plot(rng, cost_arr)
# plt.show()

