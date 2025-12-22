import random

import numpy as np
import matplotlib.pyplot as plt

# This module contains utility functions for the prediction module.

def plot_learning_curve(loss_train, loss_val, prefix="lcurve", label_loss="MSE"):
    """
    Plot the learning curve showing training and validation loss over epochs.

    Parameters
    ----------
    loss_train : list of float
        List of training loss values for each epoch.
    loss_val : list of float
        List of validation loss values for each epoch.
    prefix : str, optional
        Prefix for the saved plot filenames, by default "lcurve".
    label_loss : str, optional
        Label for the y-axis representing the type of loss, by default "MSE".
    """
    n_epochs = len(loss_train)
    # Plot training and validation loss
    plt.plot(range(1, n_epochs+1), loss_train, label="training")
    plt.plot(range(1, n_epochs+1), loss_val, label="validation")
    plt.xlabel("Epochs")
    plt.ylabel(f"{label_loss}")
    plt.legend()
    # Save the plot in both PNG and EPS formats
    plt.savefig(prefix + ".png")
    plt.savefig(prefix + ".eps")
    plt.close()

def split_data_by_torch(dataset, test_size=0.25, validation_split=0.0):
    """
    Split a dataset into training, validation, and test sets using PyTorch.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to be split.
    test_size : float, optional
        Proportion of the dataset to include in the test split, by default 0.25.
    validation_split : float, optional
        Proportion of the remaining dataset (after test split) to include in the validation split, by default 0.0.

    Returns
    -------
    tuple
        A tuple containing the training, validation, and test datasets.
    """
    import torch
    # Calculate the number of samples for each split
    n_test = int(len(dataset) * test_size)
    n_val = int((len(dataset) - n_test) * validation_split)
    n_train = len(dataset) - n_val - n_test
    # Set a fixed random seed for reproducibility
    torch.manual_seed(0)
    # Randomly split the dataset
    train, val, test = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )
    return train, val, test

def random_sampling_in_every_interval(
        loss_list, int_min, int_max, num_int, num_sample
    ):
    """
    Perform random sampling of indices within specified intervals of a loss list.

    Parameters
    ----------
    loss_list : list of float
        List of loss values.
    int_min : float
        Minimum value of the interval range.
    int_max : float
        Maximum value of the interval range.
    num_int : int
        Number of intervals to divide the range into.
    num_sample : int
        Number of samples to draw from each interval.

    Returns
    -------
    list of list
        A list of lists containing sampled indices for each interval.
    """
    # Create bins for the specified interval range
    bins = np.linspace(int_min, int_max, num_int + 1)
    # Digitize the loss list into the bins
    idxs_bins_list = np.digitize(np.array(loss_list), bins=bins)
    rep_idxs = []
    # Sample indices from each interval
    for i in range(1, num_int + 1):
        idxs = np.where(idxs_bins_list == i)[0].tolist()
        if len(idxs) == 0:
            # If no indices in the interval, append an empty list
            rep_idxs.append([])
        elif len(idxs) < num_sample:
            # If fewer indices than samples needed, sample all available indices
            rep_idxs.append(random.sample(idxs, len(idxs)))
        else:
            # Otherwise, sample the specified number of indices
            rep_idxs.append(random.sample(idxs, num_sample))
    return rep_idxs

def get_original_indices(path, data_idx, nwan):
    """
    Retrieve the original indices of materials and Wannier functions.

    Parameters
    ----------
    path : str
        Path to the directory containing the mp-id.npy file.
    data_idx : list of int
        List of data indices.
    nwan : int
        Number of Wannier functions per material.

    Returns
    -------
    tuple
        A tuple containing lists of material indices and Wannier indices.
    """
    n_wan = 10
    mp_id_path = "/home/Public/data/nx32"
    # Load material IDs from the specified path
    mp_ids = np.load(mp_id_path + "/mp-id.npy")
    # Calculate material and Wannier indices
    mp_idx = [int(mp_ids[x // n_wan]) for x in data_idx]    
    wn_idx = [x % n_wan for x in data_idx]
    return mp_idx, wn_idx

def read_loss(file_loss):
    """
    Read loss information from a file.

    Parameters
    ----------
    file_loss : str
        Path to the loss information file.

    Returns
    -------
    tuple
        A tuple containing lists of data indices and corresponding loss values.
    """
    data_idxs = []
    losses = []
    with open(file_loss) as f:
        f.readline()  # Skip the header line
        num_wan = int(f.readline())  # Read the number of Wannier functions
        for i in range(num_wan):
            data_idx, loss = f.readline().split()
            data_idxs.append(int(data_idx))
            losses.append(float(loss))
    return data_idxs, losses

def get_wannier_data(path_data):
    """
    Retrieve Wannier data and their corresponding indices.

    Parameters
    ----------
    path_data : str
        Path to the directory containing the images_center.npy file.

    Returns
    -------
    tuple
        A tuple containing lists of data indices and Wannier data arrays.
    """
    # Load all Wannier data from the specified path
    all_data = np.load(path_data + "/images_center.npy")
    nwan = all_data.shape[1]
    data_idx = []
    wannier_data = []
    for i, material in enumerate(all_data):
        for j, wannier in enumerate(material):
            # Exclude blank images
            if np.all(wannier == 0.0):
                continue
            # Append the index and data for non-blank images
            data_idx.append(i*nwan + j)
            wannier_data.append(wannier)
    return data_idx, wannier_data

def get_original_index(path_data, data_idx, nwan):
    """
    Retrieve the original indices of materials and Wannier functions.

    Parameters
    ----------
    path_data : str
        Path to the directory containing the mp-id.npy file.
    data_idx : list of int
        List of data indices.
    nwan : int
        Number of Wannier functions per material.

    Returns
    -------
    tuple
        A tuple containing lists of material indices and Wannier indices.
    """
    # Load material IDs from the specified path
    mp_all = np.load(path_data + "/mp-id.npy")
    # Calculate material and Wannier indices
    mp_idx = [int(mp_all[x // nwan]) for x in data_idx]
    wn_idx = [x % nwan for x in data_idx]
    return mp_idx, wn_idx

def read_waninfo(file_waninfo):
    """
    Read Wannier information from a file.

    Parameters
    ----------
    file_waninfo : str
        Path to the waninfo.dat file.

    Returns
    -------
    tuple
        A tuple containing lists of data indices, material indices, Wannier indices,
        loss values, and spread values.
    """
    data_idx = []
    mp_idx = []
    wn_idx = []
    loss_list = []
    spread_list = []
    with open(file_waninfo) as f:    
        f.readline()  # Skip the header line
        num_wan = int(f.readline())  # Read the number of Wannier functions
        for i in range(num_wan):
            idx, mp, wn, loss, spread = f.readline().split()
            data_idx.append(int(idx))
            mp_idx.append(int(mp))
            wn_idx.append(int(wn))
            loss_list.append(float(loss))
            spread_list.append(float(spread))
    return data_idx, mp_idx, wn_idx, loss_list, spread_list

def plot_hist(loss_list, int_min=0.0, int_max=0.001, bins=10):
    """
    Plot a histogram of the loss distribution for Wannier data.

    Parameters
    ----------
    loss_list : list of float
        List of loss values.
    int_min : float, optional
        Minimum value of the histogram range, by default 0.0.
    int_max : float, optional
        Maximum value of the histogram range, by default 0.001.
    bins : int, optional
        Number of bins in the histogram, by default 10.
    """
    fig = plt.figure()
    plt.style.use("seaborn-bright")
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax = fig.add_subplot(111)
    ax.set_title("MSE distribution of wannier data")
    ax.set_xlabel("Mean Squared Error")
    ax.set_ylabel("The number of Wannier data")
    # Plot the histogram
    n, bins, _ = ax.hist(
        loss_list, rwidth=0.9, color="blue", range=(int_min, int_max), bins=bins
    )
    # Calculate the center of each bin for annotation
    xs = (bins[:-1] + bins[1:])/2
    ys = map(int, n)
    # Annotate the histogram with the count of data in each bin
    for x, y in zip(xs, ys):
        ax.text(x, y + 140, str(y), horizontalalignment="center", size="small")
    # Save the histogram in both EPS and PNG formats
    plt.savefig("./loss_hist.eps")
    plt.savefig("./loss_hist.png")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    # Define the number of Wannier functions
    n_wan = 10
    # Specify the path to the loss information file
    file_loss = "./loss_info.dat"
    # Read the loss data from the file
    idx_list, loss_list = read_loss(file_loss)
    # Plot the histogram of the loss distribution
    plot_hist(loss_list)
    
    # Perform random sampling of indices within specified intervals
    rep_idxs = random_sampling_in_every_interval(
                 loss_list, 0.0, 0.001, 10, 10
               )
    # For each set of sampled indices, retrieve and print the original indices
    for rep_idx in rep_idxs:
        idxs_wandata = np.array(idx_list)[rep_idx]
        mp_idxs, wn_idxs = get_original_indices(idxs_wandata)
        print(idxs_wandata)
        print(mp_idxs)
        print(wn_idxs)
