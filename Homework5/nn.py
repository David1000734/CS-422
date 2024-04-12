import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Video
# https://www.youtube.com/watch?v=JHWqWIoac2I

# Create class for NN
class nnModel(nn.Module):
    # Input layer (784 features)
    # Hidden Layer (H1) --> Hidden Layer (H2)

    # Constructor, default values because it makes sense doing so
    def __init__(self, input = 784, h1 = 500, h2 = 300, h3 = 200, output = 10):
        super().__init__()        # Instantiate
        self.node1FC = nn.Linear(input, h1)
        self.node2FC = nn.Linear(h1, h2)
        self.node3FC = nn.Linear(h2, h3)
        self.out     = nn.Linear(h3, output)

    # Forward propagation
    def forward(self, x):
        # Rectified Linear Unit (Relu) is from 0 to Inf. RGB is from 0 to 255
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

# Get y values
y_actual = data.label

# Get x values
x_values = data.iloc[ :, 1:]

# Manual Seed, keep things consistent
torch.manual_seed(41)

# Create the model
model = nnModel()

# Change data frames into numpys
x_values = x_values.values
y_actual = y_actual.values

# Split data
x_train, x_test, y_train, y_test = train_test_split(x_values, y_actual, test_size = 0.2)

# Convert to tensors
x_train = torch.FloatTensor(x_train)
x_test  = torch.FloatTensor(x_test)

y_train = torch.LongTensor(y_train)
y_test  = torch.LongTensor(y_test)

# Find Error
error = nn.CrossEntropyLoss()

# Using Adam Optimizer, and setting learning rate (Epochs)
# parameters(): input, h1, h2, h3, and output
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

# Train model. Do 100 iterations (Epoch)
epochs = 100
loss_arr = []

for i in range(epochs):
    # Attempt to get a prediction
    y_pred = model.forward(x_train)

    # Check the loss
    loss = error(y_pred, y_train)       # pred vs. actual

    # Keep track of losses
    loss_arr.append(loss.detach().numpy())

    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    # Back Propagation, fine tune weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Graph
# plt.plot(range(epochs), loss_arr)
# plt.ylabel("loss/error")
# plt.xlabel("epoch")
# plt.show()

# Evaluate the model
with torch.no_grad():       # Turn off Back Propagation
    y_eval = model.forward(x_test)      # Test using test set
    loss = error(y_eval, y_test)

print(loss)
correct = 0
with torch.no_grad():
    for i, val in enumerate(x_test):
        y_val = model.forward(val)

        print(f'{i + 1}.) {str(y_val)}\t {y_test[i]}\t {y_val.argmax().item()}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1
print(f'We got {correct} correct!')
