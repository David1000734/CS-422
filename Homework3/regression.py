import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rand   # Random number
import math             # Using round function
import statistics       # Using median function
from sklearn.linear_model import LinearRegression
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

# field_name_data = ["mpg", "cylinders", "displacement",
#           "horsepower", "weight", "acceleration",
#           "model_year", "origin", "carname"]
data = pd.read_csv("auto-mpg.data.csv")

#data = pd.read_fwf("auto-mpg.data", names = field_name)

# Storing the global min and max of the mpg for later
mpg_Max = 0
mpg_Min = 0

# Hold the final result
ten_fold_table = pd.DataFrame()

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

def un_normalize_Data(data, max = mpg_Max, min = mpg_Min):
    """
    Given that we know what the specific data's min and max is, 
    we can get it's original value by the following formula:
    X: is norm-val
    Un-normalized val = X * (max - min) + min

    :param data: Normalized data point
    :param max: Original Max value of data
    :param min: Original Min value of data
    :return: coefficients of x-features. Ignoring the last column
    """

    return ((data * (max - min)) + min)

# *************** CHANGE THE CODE DOWN HERE ***************
def linear_Model(x_val, y_target, learning_rate, iteration):
    cost_arr = []
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
        # cost_arr.append(un_normalize_Data(cost, mpg_Max, mpg_Min))        # DEBUG
        # if (i % 1000 == 0):
        #     print(un_normalize_Data(cost, mpg_Max, mpg_Min))     # DEBUG)


    # DEBUG
    # rng = np.arange(0, iteration)
    # plt.plot(rng, cost_arr)
    # plt.show()
    # DEBUG
    return theta
# *************** CHANGE THE CODE UP HERE ***************


# CURRENTLY, Function likely can only return 1 row at max.
# It should be able to return up to how every many rows are
# needed for the array
def get_DupRows(source_arr, folds, multip, cur_Iter):
    temp_I = 0
    # Since mod is not 0, fill in missing rows
    rem = len(source_arr) % folds

    # Iterate however many times we need to fill in rows
    for rand_I in range(rem, (folds - 1)):
        while (temp_I == cur_Iter):
            # Rand range between 0 and 10, NOT including 10
            temp_I = rand.randrange(0, 10)
            # Keep doing random numbers until one not inside
            # the test set is picked.
        # While, END

        # Find random val within range
        randRow = rand.randrange(temp_I * multip,
                                 (temp_I + 1) * multip)
    # For, END

    # return the row
    return source_arr.iloc[randRow : randRow + 1 , : ]

def pred_values(coefficients, actual_val_Row):
    predicted_Value = 0
    mul = 0

    for i in range(len(actual_val_Row)):
        mul = (coefficients[i] * actual_val_Row.iloc[i])
        predicted_Value += mul

    return predicted_Value

def k_fold_Test(arr, folds):
    global ten_fold_table             # Build final table
    isDivisible = True
    train_arr = pd.DataFrame()        # Val to train the prediction
    test_arr = pd.DataFrame()         # Predicted values from training
    multip = 0                        # Used to find what rows to get
    max_coef = pd.DataFrame()         # Store best coefficients
    margin_error = 3                  # If value lands within #, count as valid
    coef_DR = pd.DataFrame()          # Building table

    if (folds < 2 or folds > 100):
        print("Error: K-Folds must be higher than 2 and less than 100.")
        return
    # If, END

    # To Ensure we are not geting from test, we must
    # do the getting inside the for loop
    if (len(arr) % folds != 0):
        isDivisible = False
        multip = len(arr) / folds
    else :
        multip = len(arr) / folds
        multip = math.ceil(multip)
    # if else, END
    multip = int(multip)

    # Loop number of folds
    for fold_I in range(folds):
        coef = pd.DataFrame()       # Set of Coefficients
        start_Block = fold_I * multip
        end_Block = (fold_I + 1) * multip

        # Separate data
        # Get rows of data via fold_I and multiplier. use .iloc
        if (fold_I == 0):
            test_arr = arr.iloc[start_Block : end_Block, :]
            train_arr = arr.iloc[end_Block : , :]
        else: 
            # For all other I values, the test data is inbetween train data
            test_arr = arr.iloc[start_Block : end_Block, :]

            # Get first set of data
            train_arr = arr.iloc[ : start_Block, :]
            # Get second half
            train_arr = pd.concat([train_arr, arr.iloc[end_Block : , :]],
                                  ignore_index = True)


        # Find if the folds is divisible by the array.
        # If it is, great, if not, fill in missing by randomly
        # getting columns from the array. Ensure not to grab from test
        if (not isDivisible):
            # Fill in for test_arr
            temp_Row = get_DupRows(arr, folds, multip, fold_I)
            test_arr = pd.concat([test_arr, temp_Row], ignore_index = True)
            test_arr.reset_index()

                # CURRENTLY, Does not account for potentially grabbing
                # from the test set, Light Cheating, Should fix...

            # Do the same for train_arr
            temp_Row = get_DupRows(arr, folds, multip, fold_I)
            train_arr = pd.concat([train_arr, temp_Row], ignore_index = True)
            train_arr.reset_index()
        # If notDivisible, END

        # Data processing is complete, now do the testing

        # For Train Data
        # Get X and Y
        train_X = train_arr.iloc[ :, 1 :]
        train_Y = train_arr.iloc[ :, : 1]
        found_val = 0           # Predicted Value
        act_val = 0             # Actual Value
        highest_accur = 0
        correct_count = 0
        rmse = 0                # Root Mean Square Error
        coef_series = []

        coef = linear_Model(train_X, train_Y, 0.0005, 10000)
        # Check predicted value with test array
        for i in range(len(test_arr)):          # Only check however many is in my test array
            # Only give it the features from the test array
            found_val = pred_values(coef, test_arr.iloc[ i, 1:])
            # Un_normalize
            found_val = un_normalize_Data(found_val, mpg_Max, mpg_Min)

            # Compare with actual value
            act_val = un_normalize_Data(test_arr.iloc[i, 0], mpg_Max, mpg_Min)
            print(found_val)        # DEBUG
            print(act_val)          # DEBUG
            if (found_val > act_val - margin_error and
                found_val < act_val + margin_error):
                correct_count += 1

            if ((correct_count / len(test_arr)) > highest_accur):
                max_coef = coef
                highest_accur = correct_count / len(test_arr)
        # For, END
        # Root Mean Square Error
        # Sqrt(sum from 1 to N(actual - predict)^2))
        rmse = sum((act_val - found_val) ** 2)
        rmse = round(math.sqrt(rmse), 2)
        coef = coef.reshape(1, 6)
        print(coef)

        # Un-standardize all data
        for i in range(len(data.columns)):
            # Skip mpg as coef does not have it
            if (i == 0):
                continue
            # To un-norm, we have to know the min and max of
            # each respective column.
            un_norm_data = round(un_normalize_Data(coef[0][i - 1],
                                          max(data.iloc[ :, i]),
                                          min(data.iloc[ :, i])))
            coef_series.append(un_norm_data)
        # For, END
        
        # There is no "origin" in the .csv file so none will be added
        coef_series.append(None)

        # add the Root mean Squared Error
        coef_series.append(rmse)
        
        coef_series = np.reshape(coef_series, (1, 8))
        #coef_DR = pd.DataFrame(coef_series, columns = field_name)
        ten_fold_table = pd.concat([ten_fold_table,
                                   pd.DataFrame(coef_series, columns = field_name)],
                                   ignore_index = True)
        ten_fold_table = ten_fold_table.rename(index = temp_dict)

        print(coef)
        print(rmse)
        print(ten_fold_table)
        # Clear both arrays
        test_arr = test_arr.iloc[0 : 0]
        train_arr = train_arr.iloc[0 : 0]
    # For k-folds, END

    return max_coef, round((highest_accur * 100), 2)
# Funct K_Fold_Test, END 

# ******************** Main ********************

# names, WITHOUT mpg and car name
field_name = ["cylinders", "displacement",
          "horsepower", "weight", "acceleration",
          "model_year", "origin", "RMSE"]
lr_lambda = 0.5               # Learning Rate
max_Iteration = 10000         # Iterations
k_folds = 10                  # Define fold numbers

temp_dict = {}
# Set up row names
for i in range(k_folds):
    temp_name = "Fold %i" % (i + 1)
    temp_dict[i] = temp_name

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
x = build_DF.iloc[ : , 1 : ]

# Y is the MPG, the predicting value
y = build_DF.iloc[ : , 0 : 1]

# Adding a column of 1's
#x["theta_0"] = 1

#reg = LinearRegression().fit(x, y)

# print(reg.score(x,y))
# print(reg.coef_)
# print(sum(reg.coef_))
# print(reg.intercept_)

# print(reg.predict(np.array([[0.5, 0.6, 0.4, 0.9, 1, 0.5, 0.1]])))

print(k_fold_Test(build_DF, k_folds))


print(ten_fold_table)
# predict = linear_Model(x, y, lr_lambda, max_Iteration)
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
