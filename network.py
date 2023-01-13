# Import packaages
from tqdm import tqdm
import numpy as np
import data

def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
        Value after applying sigmoid (z from the slides).
    """
    
    # Return the sigmoid of a
    return np.nan_to_num(1 / (1 + np.exp(-a)))

def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """

    # Get the row wise sum of a
    row_sum = np.sum(np.exp(a), axis=1)

    # Return the softmax of a
    return np.nan_to_num(np.exp(a) / row_sum[:, np.newaxis])

def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """

    # Return the binary cross entropy of y and t
    return ((-np.dot(t, np.log(y + 1e-30)) - np.dot((1 - t), np.log(1 - y + 1e-30)))/y.shape[0])[0]

def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """

    # Get the one hot encoded version of t
    t = data.onehot_encode(t)
    
    # Return the multiclass cross entropy of y and t
    return (np.sum(-np.sum(t * np.log(y + 1e-30), axis=1)))/y.shape[0]

class Network:
    """
    A neural network.

    Attributes
    ----------
    hyperparameters : dict
        Dictionary containing the following keys:
        model : str
            The model to use. Either 'logistic' or 'softmax'
        index1 : int
            The first index to use for logistic regression
        index2 : int    
            The second index to use for logistic regression
        rate : float
            The learning rate
        batch_size : int
            The batch size
        epochs : int
            The number of epochs
        k : int
            The number of folds for cross validation
        norm : function
            The normalization function to use
    weights : np.ndarray
        The weights of the network
    X : np.ndarray
        The training data
    y : np.ndarray
        The training labels
    Xtest : np.ndarray
        The test data
    ytest : np.ndarray
        The test labels
    stats : tuple
        The mean and standard deviation of the training data
    is_logistic : bool
        Whether the network is a logistic regression network
    activation : function
        The activation function to use
    loss : function
        The loss function to use
    norm : function
        The normalization function to use
    rate : float
        The learning rate
    batch_size : int
        The batch size
    epochs : int
        The number of epochs
    k : int
        The number of folds for cross validation

    Methods
    -------
    cross_validation()
        Perform cross validation on the training data.
    train()
        Train the network.
    test()
        Test the network.
    forward(X)
        Forward propagate the data through the network using the weights and the activation function.
    accuracy(y, t)
        Compute the accuracy of the network.
    """

    def __init__(self, hyperparameters):
        """
        Initialize the network.

        Parameters
        ----------
        hyperparameters : dict
            Dictionary containing the following keys:
            model : str
                The model to use. Either 'logistic' or 'softmax'
            index1 : int
                The first index to use for logistic regression
            index2 : int
                The second index to use for logistic regression
            rate : float
                The learning rate
            batch_size : int
                The batch size
            epochs : int
                The number of epochs
            k : int
                The number of folds for cross validation
            norm : function
                The normalization function to use
            
        Returns
        -------
        None
        """
        
        # Set the hyperparameters
        self.hyperparameters = hyperparameters

        # Set the norm and bool indicator
        self.is_logistic = (hyperparameters.model == 'logistic')
        self.norm = hyperparameters.norm

        # Set the weights
        self.weights = np.zeros((32*32+1, 1 if self.is_logistic else 10))

        # Load the data
        self.X, self.y = data.load_data()
        self.Xtest, self.ytest = data.load_data(train = False)

        # Pick the class if logistic
        if self.is_logistic:
            # Get the indices
            mask = (self.y == hyperparameters.index1) | (self.y == hyperparameters.index2)

            # Set the data
            self.X = self.X[mask]
            self.y = self.y[mask]

            # Set the labels
            self.y[self.y == hyperparameters.index1] = 0
            self.y[self.y == hyperparameters.index2] = 1

            # Get the indices
            masktest = (self.ytest == hyperparameters.index1) | (self.ytest == hyperparameters.index2)

            # Set the data
            self.Xtest = self.Xtest[masktest]
            self.ytest = self.ytest[masktest]

            # Set the labels
            self.ytest[self.ytest == hyperparameters.index1] = 0
            self.ytest[self.ytest == hyperparameters.index2] = 1

        # Normalize the data
        self.X, self.stats = self.norm(self.X)
        self.Xtest, _ = self.norm(self.Xtest, self.stats[0], self.stats[1])

        # Apply bias
        self.X = data.append_bias(self.X)
        self.Xtest = data.append_bias(self.Xtest)

        # Set the rate, batch size, epochs, and k
        self.rate = hyperparameters.rate
        self.batch_size = hyperparameters.batch_size
        self.epochs = hyperparameters.epochs
        self.k = hyperparameters.k

        # Set the activation and loss functions
        self.activation = sigmoid if self.is_logistic else softmax
        self.loss = binary_cross_entropy if self.is_logistic else multiclass_cross_entropy
    
    def cross_validation(self, ealry_stopping = False):
        """
        Perform cross validation on the training data.
        
        Use `self.train' to train the network on each fold.
        Use `self.test' to test the network on each fold.
        Save the loss and accuracy of each fold.

        Returns
        -------
        loss : list
            The loss of each fold
        accuracy : list
            The accuracy of each fold
        testloss : list
            The test loss of each fold
        testaccuracy : list
            The test accuracy of each fold
        best_weights : np.ndarray
            The weights of the network with the best test accuracy
        """

        # Set the loss and accuracy list to store the loss and accuracy for each fold
        loss = np.array([])
        accuracy = np.array([])

        # Set the validation loss and accuracy list to store the validation loss and accuracy for each fold
        testloss = np.array([])
        testaccuracy = np.array([])

        # Split the data into k folds
        train_X = np.array_split(self.X, self.k)
        train_y = np.array_split(self.y, self.k)

        # Iterate over the folds
        for i in range(self.k):
            # Set the training data
            X = np.concatenate(train_X[:i] + train_X[i+1:])
            y = np.concatenate(train_y[:i] + train_y[i+1:])

            # Initialize the weights
            self.weights = np.zeros((32*32+1, 1 if self.is_logistic else 10))

            # Store the loss and accuracy for this training fold
            l, a, l_val, a_val, _ = self.train((X, y), (train_X[i], train_y[i]), ealry_stopping)
            loss = np.append(loss, l)
            accuracy = np.append(accuracy, a)
            testloss = np.append(testloss, l_val)
            testaccuracy = np.append(testaccuracy, a_val)

        # Return the loss, accuracy, validation loss, validation accuracy
        return loss, accuracy, testloss, testaccuracy

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """

        # Return the activation of the product of the weights and X
        return self.activation(np.dot(X, self.weights))

    def train(self, minibatch, validation, early_stopping = False):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        loss : list
            The loss of each epoch
        accuracy : list
            The accuracy of each epoch
        testloss : list
            The test loss of each epoch
        testaccuracy : list
            The test accuracy of each epoch
        best_weights : np.ndarray
            The weights of the network with the best test accuracy
        """

        # Get the X and y from the minibatch
        X, t = minibatch

        # Create the loss and accuracy array
        loss = np.array([])
        accuracy = np.array([])

        # Create the loss and accuracy array for validation
        loss_val = np.array([])
        accuracy_val = np.array([])

        # Create the minimum loss
        min_loss = np.inf

        # Create the index for early stopping
        index = 0 

        # Initialize the weights
        self.weights = np.zeros((32*32+1, 1 if self.is_logistic else 10))
        best_weights = None

        # Iterate over the epochs
        for i in tqdm(range(self.epochs)):
            # Shuffle the data
            X, t = data.shuffle((X, t))

            # Iterate over the batches
            for j in range(0, len(X), self.batch_size):
                # Get the batch
                X_batch = X[j:j+self.batch_size]
                t_batch = t[j:j+self.batch_size]

                # Compute the output
                y = self.forward(X_batch)

                # Stochastic gradient descent process for logistic regression
                if self.is_logistic:
                    # Compute the gradient and update the weights
                    self.weights += self.rate * np.sum(np.dot(X_batch.T, (t_batch - y)), axis=1).reshape(-1, 1)

                # Stochastic gradient descent process for softmax regression
                else:
                    # Compute the gradient and update the weights
                    self.weights += self.rate * np.dot(X_batch.T, (data.onehot_encode(t_batch) - y))
            
            # Compute the loss and accuracy for this epoch and store it
            l, a = self.test(minibatch)
            l_val, a_val = self.test(validation)

            # Append the loss and accuracy
            loss_val = np.append(loss_val, l_val)
            accuracy_val = np.append(accuracy_val, a_val)
            loss = np.append(loss, l)
            accuracy = np.append(accuracy, a)

            # Early stopping
            if early_stopping:
                # Check if the loss is lower than the minimum loss
                if l_val < min_loss:
                    # Update the minimum loss
                    index = i
                    min_loss = l_val
                    best_weights = self.weights

                # Check if the loss is higher than the minimum loss
                elif i - index > 10:
                    # Save the best weights
                    self.weights = best_weights

                    # Break the loop
                    break

        # If use early stopping, return the best loss
        if early_stopping:
            return None, None, np.array([min_loss]), None, None

        # Return the loss and accuracy
        return loss, accuracy, loss_val, accuracy_val, self.weights

    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """

        # Get the X and y from the minibatch
        X, y = minibatch
        
        # Compute the loss and accuracy of the network and store it
        loss = self.loss(self.forward(X), y)
        accuracy = self.accuracy(self.forward(X), y)

        # Return the loss and accuracy
        return loss, accuracy
    
    def accuracy(self, y, t):
        """
        Compute the accuracy of the network's predictions.

        Parameters
        ----------
        y
            The network's predictions
        t
            The corresponding targets
        Returns
        -------
        float
            accuracy of the network's predictions
        """

        # Data processing for logistic regression
        if self.is_logistic:
            # Get the predicted class
            y = np.where(y >= 0.5, 1, 0)

            # Flatten the arrays
            y = y.flatten()
            t = t.flatten()
        
        # Data processing for multiclass classification
        else:
            # Get the predicted class
            y = np.argmax(y, axis=1)
        
        # Return the accuracy
        return np.mean(y == t)