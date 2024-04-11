import pandas as pd
from sklearn.svm import SVC                 # Get SVM model
from sklearn.model_selection import KFold   # K-fold test

# Apply SVM with three kernels
# linear, poly, and rbf. Additionally,
# compute accuracy using 5-fold CV

# ******************** Main ********************
# Reading data from csv, specifying header is at the 0'th position
data = pd.read_csv("MNIST_HW4.csv", header = [0])

# Pull the label with all the numbers out
y_actual = data.label
# Take only the values from the array out
x_values = data.iloc[ :, 1 :]
# Pick X train and test values later in the code.

'''
    ********** Recall: **********
    * C:     Regularization. Specifies how specific the drawn line should be.
                A low value indicates a more general line while a higher one
                creates a line better at separating each different value.
            Low  Regularization: Some miss-classification may occur. However,
                                 this may be a preferred outcome.
            High Regularization:

    * Gamma: Specifies if only points near the line should be considered or if
                values farther away should be considered as well. 
                Low gamma indicates farther points are considered while
                high gamma indicates only closer points are. 
            High Gamma: 
            Low  Gamma: May result in lower accuracy. May have issue with accuracy.
                        However, may be okay as it could be more
                        computationally efficient. 

    * Kernel: In cases where data sets appear to not have a clear pattern, changing
                the dimension to a higher one, finding a pattern, and then
                returning to the original dimension would be a solution.
        z = x^2 + y^2, Where z is called the kernel
'''

# Create k-fold test for 5 folds
kf = KFold(n_splits = 5, shuffle = True)

# Accuracy of each test, used to hold max
linear_acc = 0
poly_acc   = 0
rbf_acc    = 0

idx = 1     # counter
# For these n_splits occurrences, get a new set of train and test
for train_idx, test_idx in kf.split(x_values): 
    # train_idx will contain training split and
    # test_idx will contain it's respective test set
    print("\t\tFold %i: " % idx)
    curr_acc = 0            # Used to find highest accuracy among tests

    # *** These are only the indexes of the array, must access
    #     the array to get actual values. ***
    x_tr  = x_values.iloc[train_idx]
    x_tst = x_values.iloc[test_idx]
    y_tr  = y_actual.iloc[train_idx]
    y_tst = y_actual.iloc[test_idx]
    # Set respective variables from the array

    # Try method with Linear kernel
    newModel = SVC(kernel = 'linear')       # Create SVM
    newModel.fit(x_tr, y_tr)                # Train Model with fit method
    # Check accuracy with test values and print it
    curr_acc = newModel.score(x_tst, y_tst)
    print("Linear: %.4f" % (curr_acc * 100))

    # Find highest accuracy of the linear model
    if (curr_acc > linear_acc): 
        linear_acc = curr_acc

    # Try method with poly kernel
    newModel = SVC(kernel = 'poly')
    newModel.fit(x_tr, y_tr)
    curr_acc = newModel.score(x_tst, y_tst)
    print("Poly: %.4f" % (curr_acc * 100))

    if (curr_acc > poly_acc): 
        poly_acc = curr_acc

    # try method with rbf kernel
    newModel = SVC(kernel = 'rbf')
    newModel.fit(x_tr, y_tr)
    curr_acc = newModel.score(x_tst, y_tst)
    print("RBF: %.4f" % (curr_acc * 100))

    if (curr_acc > rbf_acc): 
        rbf_acc = curr_acc

    idx += 1        # incrementer
# For, END
# NOTE: Not changing gamma or regularization resulted in the highest accuracy.

# Print highest accuracy
print("\n\t\t Highest Accuracy result from K-folds")
print("Highest accuracy of Linear Model was: %.2f" % (linear_acc * 100))
print("Highest accuracy of Poly Model was: %.2f"   % (poly_acc * 100))
print("Highest accuracy of RBF Model was: %.2f"    % (rbf_acc * 100))
