import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import local modules using relative imports
try:
    from ...src.hiap.prediction import utils
    from ...src.hiap.prediction import datasets
except ImportError:
    # Fallback for when running as standalone script
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from hiap.prediction import utils
    from hiap.prediction import datasets

# Define constants for the model and data
size = 10.0  # Size parameter for the dataset
l_max = 3  # Maximum angular momentum
n_c = (l_max + 1)**2  # Number of coefficients
n_div = 32  # Number of divisions
rn_max = 4  # Maximum radial number
n_samples = 20000  # Number of samples for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device configuration

class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with three fully connected layers.
    """
    def __init__(self, input_size, output_size):
        """
        Initialize the neural network layers.

        Parameters
        ----------
        input_size : int
            The size of the input features.
        output_size : int
            The size of the output features.
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 10*output_size)  # First fully connected layer
        self.fc2 = nn.Linear(10*output_size, 5*output_size)  # Second fully connected layer
        self.fc3 = nn.Linear(5*output_size, output_size)  # Third fully connected layer

    def forward(self, x):
        """
        Define the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the network.
        """
        x = F.elu(self.fc1(x))  # Apply ELU activation after first layer
        x = F.elu(self.fc2(x))  # Apply ELU activation after second layer
        x = self.fc3(x)  # Output layer
        return x

class LossAsImage(torch.autograd.Function):
    """
    Custom loss function for the neural network.
    """
    @staticmethod
    def forward(ctx, y_pred, x_train):
        """
        Forward pass for the custom loss function.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object to save information for backward computation.
        y_pred : torch.Tensor
            Predicted output from the model.
        x_train : torch.Tensor
            Ground truth data.

        Returns
        -------
        torch.Tensor
            Mean squared error between predicted and true values.
        """
        ctx.save_for_backward(y_pred, x_train)  # Save tensors for backward pass
        y_pred = y_pred.to("cpu").detach().numpy().copy()  # Detach and copy prediction
        se = squared_error(y_pred, x_train)  # Calculate squared error
        ctx.mse_resp = torch.mean(se, dim=1)  # Mean squared error response
        return torch.mean(se)  # Return mean squared error

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the custom loss function.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object with saved tensors.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        tuple
            Gradients with respect to the inputs.
        """
        dy = 1e-4  # Small perturbation for numerical gradient
        diff = []
        y, x = ctx.saved_tensors  # Retrieve saved tensors
        y = y.to('cpu').detach().numpy().copy()  # Detach and copy prediction
        for i in range(y.shape[1]):
            y[:, i] += dy  # Perturb the prediction
            diff.append(squared_error(y, x).mean(dim=1) - ctx.mse_resp)  # Calculate difference
            y[:, i] -= dy  # Revert the perturbation
        grad_input = torch.t(torch.cat(diff).reshape(len(diff), -1)) / dy  # Calculate gradient
        return grad_input, None  # Return gradient and None for x_train

def squared_error(y_pred, x_train):
    """
    Calculate the squared error between predicted and true values.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values.
    x_train : torch.Tensor
        True values.

    Returns
    -------
    torch.Tensor
        Squared error tensor.
    """
    c, gamma, an = np.split(y_pred, [n_c, n_c+1], axis=1)  # Split predictions
    output = boxdata.params_to_boxdata(c, gamma, an)  # Convert parameters to box data
    return (torch.tensor(output).to(device) - x_train).pow(2)  # Calculate squared error

def training(n_epoch, n_batch):
    """
    Train the neural network model.

    Parameters
    ----------
    n_epoch : int
        Number of training epochs.
    n_batch : int
        Batch size for training.

    Returns
    -------
    tuple
        Lists of training and validation losses.
    """
    loss_train_list = []  # List to store training losses
    loss_val_list = []  # List to store validation losses
    train_loader = torch.utils.data.DataLoader(train, n_batch, shuffle=True)  # DataLoader for training data
    val_loader = torch.utils.data.DataLoader(val, n_batch, shuffle=True)  # DataLoader for validation data
    for epoch in range(n_epoch):
        print(f'epoch ={epoch+1:4d}/{n_epoch}')  # Print current epoch
        count = 0  # Initialize count for processed samples
        for i, x_train in enumerate(train_loader):
            x_train = x_train.to(device)  # Move data to device
            y_pred = model(x_train)  # Forward pass
            loss = criterion(y_pred, x_train)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Reset gradients
            count += int(x_train.shape[0])  # Update count
            print(f'{count:>6d}/{len(train)} : loss ={loss.item():>10.6f}')  # Print progress
        loss_train = evaluation(train_loader)  # Evaluate training loss
        loss_val = evaluation(val_loader)  # Evaluate validation loss
        print(f'final : train_loss ={loss_train.item():>10.6f} - val_loss ={loss_val.item():>10.6f}\n') 
        loss_train_list.append(loss_train)  # Append training loss
        loss_val_list.append(loss_val)  # Append validation loss
    return loss_train_list, loss_val_list  # Return loss lists

def evaluation(data_loader):
    """
    Evaluate the model on a given dataset.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        DataLoader for the dataset to evaluate.

    Returns
    -------
    np.ndarray
        Mean squared error of the model on the dataset.
    """
    with torch.no_grad():  # Disable gradient computation
        ses = []  # List to store squared errors
        for i, x in enumerate(data_loader):
            x = x.to(device)  # Move data to device
            y = model(x)  # Forward pass
            ses.append(squared_error(y.to("cpu").detach().numpy().copy(), x))  # Calculate squared error
        mse = torch.cat(ses).mean()  # Calculate mean squared error
        return mse.to("cpu").numpy().copy()  # Return mean squared error

if __name__ == "__main__":
    # Prepare the dataset
    boxdata = datasets.BoxData(l_max, rn_max, n_div, size)  # Initialize BoxData
    x_numpy, y_numpy = boxdata.generate_learning_data(n_samples)  # Generate learning data
    x_torch = torch.from_numpy(x_numpy)  # Convert numpy array to torch tensor
    train, val, test = utils.split_data_by_torch(x_torch, test_size=0.1, validation_split=0.1)  # Split data
    print(f'num_train={len(train)}, num_val={len(val)}, num_test={len(test)}')  # Print dataset sizes
    
    # Initialize the model
    model = NeuralNetwork(x_torch.shape[1], y_numpy.shape[1])  # Create model instance
    model.to(device)  # Move model to device

    # Define loss function and optimizer
    criterion = LossAsImage.apply  # Set custom loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, eps=1e-3)  # Initialize optimizer

    # Training loop
    n_batch = 128  # Batch size
    n_epoch = 300  # Number of epochs
    loss_trains, loss_vals = training(n_epoch, n_batch)  # Train the model
    utils.plot_learning_curve(loss_trains, loss_vals, prefix="encoder")  # Plot learning curve

    # Evaluate the model on the test set
    test_loader = torch.utils.data.DataLoader(test, n_batch, shuffle=True)  # DataLoader for test data
    loss_tests = evaluation(test_loader)  # Evaluate test loss
    print(f'test_loss ={loss_tests.item():>10.6f}')  # Print test loss

    # Save the trained model
    torch.save(model.state_dict(), 'encoder.pth')  # Save model state dictionary
