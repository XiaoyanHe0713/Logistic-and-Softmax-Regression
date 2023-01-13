# Import packages
import numpy as np
import pickle
import os

def load_data(train = True):
    """
    Loads dataset into python
    
    Parameters
    ----------
    train : bool
        Whether to load the training or test set

    Returns
    -------
        Tuple:
            X : np.array
                The data
            y : np.array
                The targets
    """

    # Define the function to open file
    def unpickle(filepath):
        # Open file
        with open(os.path.join('./cifar-10-batches-py', filepath), 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        
        # Return the data and the labels
        return dict_

    # Raises error if the dataset does not exist
    if not os.path.exists('./cifar-10-batches-py'):
        raise ValueError('Need to run get_data.sh before writing any code!')

    # Create the X and y arrays
    full_data = None
    full_labels = None
    
    # Store the dataset into full_date and full_labels
    batches = [f'data_batch_{i+1}' for i in range(5)] if train else ['test_batch']

    # Iterate through the batches
    for batch in batches:

        # Load the data
        dict_ = unpickle(batch)
        data = np.array(dict_[b'data'].reshape(-1, 3, 1024).mean(axis=1))
        labels = np.array(dict_[b'labels']) 

        # Concatenate the data and labels
        full_data = data if full_data is None else np.concatenate([full_data, data])
        full_labels = labels if full_labels is None else np.concatenate([full_labels, labels])

    # Return the data and labels
    return full_data, full_labels

def z_score_normalize(X, u = None, sd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    
    # Calculate the mean and standard deviatin of x, then perform normalization
    u = np.mean(X, axis=0) if u is None else u
    sd = np.std(X, axis=0) if sd is None else sd

    # Return the normalized data and the mean and standard deviation
    return np.nan_to_num(X - u) / sd, (u, sd)

def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    
    # Calculate the min and max of x, then perform normalization
    _min = np.min(X, axis=0) if _min is None else _min
    _max = np.max(X, axis=0) if _max is None else _max

    # Return the normalized data and the min and max
    return np.nan_to_num((X - _min + 1e-30) / (_max - _min + 1e-30)), (_min, _max)

def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    
    # Represent categorical data by setting the component corresponding to the target category to 1 and all others to 0
    return np.eye(10)[y]

def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    
    # Return an array of index of the largest element (1) in each row
    return np.argmax(y, axis=1)

def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """

    # Extract the X and y from the dataset
    X, y = dataset

    # Randomly permute a sequence with length of X
    order = np.random.permutation(len(X))

    # Return the shuffled X and y
    return X[order], y[order]

def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape ((N+1),d)
    """

    # Append the bias term with a constant value of 1 to the X set
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

# Split the dataset into batches
def generate_minibatches(dataset, batch_size=64):
    """
    Generate minibatches from dataset.

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)
    batch_size : int
        Size of each minibatch

    Returns
    -------
        Generator of minibatches
    """

    # Extract the X and y from the dataset
    X, y = dataset

    # Split the dataset into batches
    l_idx, r_idx = 0, batch_size

    # Generate minibatches from dataset
    while r_idx < len(X):
        # Keep updating the right index until index out of range
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    # Return the last batch
    yield X[l_idx:], y[l_idx:]

def generate_k_fold_set(dataset, k = 5): 
    """
    Generate k-fold set from dataset.

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)
    k : int
        Number of folds

    Returns
    -------
        Generator of k-fold sets
    """

    # Extract the X and y from the dataset
    X, y = dataset

    # If it is only one fold, return the whole dataset
    if k == 1:
        # Return the whole dataset
        yield (X, y), (X[len(X):], y[len(y):])
        return

    # Randomly permute a sequence with length of X
    order = np.random.permutation(len(X))
    
    # Split the dataset into k folds
    fold_width = len(X) // k

    # Generate k-fold sets from dataset
    l_idx, r_idx = 0, fold_width

    # Apply the method of k-fold cross-validation to separate a portion from training data as validation
    for i in range(k):

        # Keep updating the right index until index out of range
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate([y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]

        # Return the training and validation set
        yield train, validation

        # Update the left and right index
        l_idx, r_idx = r_idx, r_idx + fold_width
