import pandas as pd             # Reading data
import torch                    # Neural Network
import torch.nn as nn           # Neural Network, using pytorch
import torch.nn.functional as F             # Contains activation functions
from sklearn.model_selection import KFold   # K-fold test

# Create class for NN. More organized
class nnModel(nn.Module):
    '''
    Neural Network class which will take in 784 inputs and
    return 10 outputs prediction a value from 0 - 9.
    '''
    # Input layer (784 features)
    # Hidden Layer (H1) --> Hidden Layer (H2) --> ... --> output

    # Constructor, default values because it makes sense doing so
    def __init__(self, input = 784, h1 = 500, h2 = 300, h3 = 200, output = 10):
        '''
        Constructor, declared our hidden layers, input, and output.
        Will be using 784 as input, 500, 300, 200 as hidden layers,
        and 10 as output.

        params: Will follow the same format as stated above
        '''
        super().__init__()        # Instantiate
        self.node1FC = nn.Linear(input, h1)
        self.node2FC = nn.Linear(h1, h2)
        self.node3FC = nn.Linear(h2, h3)
        self.out     = nn.Linear(h3, output)

    def forward(self, x):
        '''
        Function will perform the actual testing/evaluating and use
        Rectified Linear Unit for it. As pixels range from 0 to 255 and
        RELU is from 0 to inf, it fits the model.

        param x: The test data to be trained or predicted.

        return: Output the 10 different values and the predictions
        for each value. The highest one is the predicted value.
        '''
        # Run through each of the hidden layers
        x = F.relu(self.node1FC(x))
        x = F.relu(self.node2FC(x))
        x = F.relu(self.node3FC(x))

        # Finally output.
        x = self.out(x)
        return x
    pass
# End of nnModel class

# ******************** Main ********************
# Read data in and specify header at 0th index
data = pd.read_csv("MNIST.csv", header = 0)

debug = False            # Debug variable

# Get y values
y_actual = data.label

# Get x values
x_values = data.iloc[ :, 1:]

epochs = 100            # Train Model, 100 iterations

# Define our K_fold test
kf = KFold(n_splits = 5, shuffle = True)

idx = 1         # incrementor

sum = 0         # Keep track of average...

# Create splits for k_fold
print("\t\t Starting K-Fold test.")
for train_idx, test_idx in kf.split(x_values):
    # Create the model
    model = nnModel()

    # we need the size of y. Using the idx for this is fine
    y_size = test_idx.shape[0]

    # Setup all x/y train and test
    x_tr  = x_values.iloc[train_idx].to_numpy()
    x_tst = x_values.iloc[test_idx].to_numpy()
    y_tr  = y_actual.iloc[train_idx].to_numpy()
    y_tst = y_actual.iloc[test_idx].to_numpy()
    # We will have to change to numpy for the tensors

    # Convert to Tensor
    x_tr  = torch.Tensor(x_tr)
    x_tst = torch.Tensor(x_tst)

    y_tr  = torch.LongTensor(y_tr)
    y_tst = torch.LongTensor(y_tst)

    # Find error
    error = nn.CrossEntropyLoss()

    # Using Adam Optimizer, and setting learning rate (Epochs)
    # parameters(): input, h1, h2, h3, and output
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    # Train model
    for i in range(epochs):
        # Train the model
        y_pred = model.forward(x_tr)

        # Check for loss
        loss = error(y_pred, y_tr)

        # Usefull print if needed
        if ((i % 10 == 0) and debug):
            # How is the training going? Is the loss decreasing?
            print("Epoch: %i, loss: %0.8f" % (i, loss))

        # Back Propagation, fine tune weights
        optimizer.zero_grad()       # Back Propagation
        loss.backward()
        optimizer.step()

    # Evaluate model
    with torch.no_grad():       # Turn off Back Propagation
        # Now test it without Back Propagation
        y_eval = model.forward(x_tst)
        loss = error(y_eval, y_tst)        # Calculate loss
    
    # Usefull print if needed
    if (debug):
        # Is our loss value similar to the ones calculated earlier?
        print(loss)

    correct = 0             # Count correct predictions
    # See how it does with test data
    with torch.no_grad():
        # Where i is index and val is the actual data points
        for i, val in enumerate(x_tst):
            # Once again, perform the test/evaluation
            y_val = model.forward(val)

            # Get the highest value's index. That is the predicted val
            # Check with actual. Inc if guess correct.
            if y_val.argmax().item() == y_tst[i]:
                correct += 1
        # For, END
    # With, END

    sum += (correct / y_size) * 100

    # Print Results
    print("Fold %i: %i correct with %0.2f%% accuracy.\n" % \
        (idx, correct, (correct / y_size) * 100))

    idx += 1            # Counter
# for KFold, END
print("Average accuracy of the Neural Network: %0.2f." % (sum / 5))
