import pandas as pd
import numpy as np
import math     # Needed for floor function

test_data = pd.read_csv("MNIST_test.csv", header= None)
train_data = pd.read_csv("MNIST_training.csv", header= None)
# remove labels header
train_data = train_data.iloc[ 1:, :]
test_data = test_data.iloc[ 1:, :]

print(train_data.shape)
print(test_data.shape)

# Function completes the euclidean_Distance and assumes the
# testRow parameter is a singular row.
def euclidean_Distance(testRow, trainVec) :
    # Duplicate testfile to equal size of training
    dup_test = np.tile(testRow, (len(trainVec), 1))

    # Calculate Euclidean Distance, res1 = (trainVec[k] - testRow[k]) ^ 2,
    #       where k is for all values in row
    vec_sub = (trainVec.to_numpy().astype(int) - dup_test.astype(int)) **2

    # Return sum of above for all values and then square root
    return (np.sqrt(np.sum(vec_sub, axis= 1)))
# End Of Function

# Not a true function, here for convinience.
def KNN_Funct() :
    guess_Correct = 0
    guess_Wrong = 0

    # For loop to check entire test_data
    for i in range(len(test_data)):
        group_arr = [0] * 10        # Create size 10 int array

        # retrieve the i'th row of every iteration
        dist_Row = euclidean_Distance(test_data.iloc[i, :], train_data)

        # Prepare to find max, first find max index
        dist_sorted_index = np.argsort(dist_Row)
        # use max index to retrieve max
        dist_sorted_arr = dist_Row[dist_sorted_index]

        # Search Max array up to the k'th element
        for x in dist_sorted_arr[ : k]:
            # Find the index of that element from the original data (row)
            temp = np.where(x == dist_Row)

            # Division to find what group it belongs to.
            # Data is separated with by 95 size blocks of data
            group_arr[math.floor(temp[0][0] / 95)] += 1
        # End Inner For

        # In test data, data is separated by blocks of 5
        actualVal = math.floor(i / 5)
        # From the guess array, find the largest value, that is our guess
        guessVal = group_arr.index(max(group_arr))

        # Check if guess matches with actual
        if (actualVal == guessVal):
            guess_Correct += 1
        else:
            guess_Wrong += 1
            print("Incorrect Guess. \nActual value: %i, Guessed Value: %i\n" % (actualVal, guessVal))
    # End Outer For

    # Calculate accuracy, total test data is 50
    accuracy = (guess_Correct / 50) * 100
    print("Accuracy of KNN with K = %i: %i%%\n" % (k, accuracy))
# End of Function

# ********** Using KNN **********
k = 1

while ((int(k)) > 0):
    print("Enter 0 to exit.")

    # User Input of K, no upper bound so uh...
    k = input("Please enter a K value: ")

    try:
        # Check if cast is valid
        if ((int(k)) and (int(k)) > 0):
            k = (int(k))    # Input is taken as string, cast to int

            KNN_Funct()     # Call to main function
        # Cast WILL be VALID but value is smaller than 0
        else:
            print("Error: K value must be greater than 0.")

    # If cast is not valid, enter except state.
    except:
        print("Error: K value is non-numeric.")
        k = 0           # End Loop


