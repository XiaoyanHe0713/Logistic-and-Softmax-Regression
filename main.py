# Import packages
import matplotlib.pyplot as plt
from PIL import Image as im
from matplotlib import cm
import numpy as np
import argparse
import network
import pickle
import data
import os

def run_single(hyperparameters):
        """
        Run a single cross validation based on the hyperparameters

        Parameters
        ----------
        hyperparameters : argparse.Namespace
                hyperparameters for the network

        Returns
        -------
        best_weight: np.array
                the best weight for the network
        """

        # Build the network with the given hyperparameters
        net = network.Network(hyperparameters)

        # Store the cross validation result into variables
        loss, accuracy, testloss, testaccuracy = net.cross_validation()

        # Get the average accuracy for all the folds
        accuracy = np.reshape(accuracy, (hyperparameters.k, hyperparameters.epochs))
        accuracy = np.mean(accuracy, axis = 0)

        # Get the average loss for all the folds
        loss = np.reshape(loss, (hyperparameters.k, hyperparameters.epochs))
        loss = np.mean(loss, axis = 0)

        # Get the average test accuracy for all the folds
        testaccuracy = np.reshape(testaccuracy, (hyperparameters.k, hyperparameters.epochs))
        testaccuracy = np.mean(testaccuracy, axis = 0)

        # Get the average test loss for all the folds
        testloss = np.reshape(testloss, (hyperparameters.k, hyperparameters.epochs))
        testloss = np.mean(testloss, axis = 0)

        # Get the index for graphing
        index_avg = np.arange(hyperparameters.epochs)
        index_avg += 1

        # Create a plot for training loss
        loss_train_plt = plt.figure()

        # Fit line for training loss
        loss_poly = np.polyfit(index_avg, loss, 7)
        loss_poly_y = np.poly1d(loss_poly)(index_avg)
        loss_test_poly = np.polyfit(index_avg, testloss, 7)
        loss_test_poly_y = np.poly1d(loss_test_poly)(index_avg)

        # Calculate the early stop point
        early_stop = np.argmin(testloss)
        early_stop_loss = testloss[early_stop]
        early_stop_accuracy = testaccuracy[early_stop]
        early_stop += 1

        # Plot both raw and fited line
        plt.plot(index_avg, loss, label = 'Training Loss Average', color = 'blue')
        plt.plot(index_avg, loss_poly_y, label = 'Training Loss Average Fitted', color = 'aqua')
        plt.plot(index_avg, testloss, label = 'Validation Loss Average', color = 'red')
        plt.plot(index_avg, loss_test_poly_y, label = 'Validation Loss Average Fitted', color = 'hotpink')

        # Plot the early stop line
        plt.axvline(x = early_stop, label = 'Minimum Loss Epoch', color = 'green')

        # Plot the point on the early stop line
        loss_str = "Minimum Validation Loss: " + str(early_stop_loss)
        plt.plot(early_stop, early_stop_loss, 'o', color = 'green', label = loss_str)

        # Set the size of the plot
        plt.gcf().set_size_inches(10, 7)

        # Set the title and other aspect of the plot
        plt.title('Cross-Validation Loss Average (No Early Stop)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Save the plot
        loss_train_plt.savefig('cv_loss.png')

        # Create a plot for training accuracy
        accuracy_train_plt = plt.figure()

        # Fit line for training accuracy
        acc_poly = np.polyfit(index_avg, accuracy, 7)
        acc_poly_y = np.poly1d(acc_poly)(index_avg)
        acc_test_poly = np.polyfit(index_avg, testaccuracy, 7)
        acc_test_poly_y = np.poly1d(acc_test_poly)(index_avg)

        # Plot both raw and fited line
        plt.plot(index_avg, accuracy, label = 'Training Accuracy Average', color = 'blue')
        plt.plot(index_avg, acc_poly_y, label = 'Training Accuracy Average Fitted', color = 'aqua')
        plt.plot(index_avg, testaccuracy, label = 'Validation Accuracy Average', color = 'red')
        plt.plot(index_avg, acc_test_poly_y, label = 'Validation Accuracy Average Fitted', color = 'pink')

        # Plot the early stop line
        plt.axvline(x = early_stop, label = 'Minimum Loss Epoch', color = 'green')

        # Plot the point on the early stop line
        acc_str = "Validation Accuracy at Minimum Validation Loss: " + str(early_stop_accuracy)
        plt.plot(early_stop, early_stop_accuracy, 'o', color = 'green', label = acc_str)

        # Set the size of the plot
        plt.gcf().set_size_inches(10, 7)
        
        # Set the title and other aspect of the plot
        plt.title('Cross-Validation Accuracy Average (No Early Stop)')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Save the plot
        accuracy_train_plt.savefig('cv_accuracy.png')

        # Single trainning
        loss, accuracy, _, _, weights = net.train((net.X, net.y), (net.Xtest, net.ytest))

        # Test the model
        test_loss, test_accuracy = net.test((net.Xtest, net.ytest))
        
        # Get the index for graphing
        index_plot = np.arange(len(loss))
        index_plot += 1

        # Create a plot for training loss
        loss_train_plt = plt.figure()

        # Fit line for training loss
        loss_poly = np.polyfit(index_plot, loss, 7)
        loss_poly_y = np.poly1d(loss_poly)(index_plot)

        # Plot both raw and fited line
        plt.plot(index_plot, loss, label = 'Whole Dataset Training Loss', color = 'blue')
        plt.plot(index_plot, loss_poly_y, label = 'Whole Dataset Training Loss Fitted', color = 'aqua')

        # Plot the test loss line
        plt.axhline(y = test_loss, label = 'Test Dataset Loss', color = 'red')

        # Plot the point on the test loss line
        test_loss_str = "Test Dataset Loss: " + str(test_loss)
        plt.plot(net.epochs, test_loss, 'o', color = 'red', label = test_loss_str)

        # Set the size of the plot
        plt.gcf().set_size_inches(10, 7)

        # Set the title and other aspect of the plot
        plt.title('Whole Dataset Training Loss and Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Save the plot
        loss_train_plt.savefig('final_loss.png')

        # Create a plot for training accuracy
        accuracy_train_plt = plt.figure()

        # Fit line for training accuracy
        acc_poly = np.polyfit(index_plot, accuracy, 7)
        acc_poly_y = np.poly1d(acc_poly)(index_plot)

        # Plot both raw and fited line
        plt.plot(index_plot, accuracy, label = 'Whole Dataset Training Accuracy', color = 'blue')
        plt.plot(index_plot, acc_poly_y, label = 'Whole Dataset Training Accuracy Fitted', color = 'aqua')

        # Plot the test accuracy line
        plt.axhline(y = test_accuracy, label = 'Test Dataset Accuracy', color = 'red')

        # Plot the point on the test accuracy line
        test_acc_str = "Test Dataset Accuracy: " + str(test_accuracy)
        plt.plot(net.epochs, test_accuracy, 'o', color = 'red', label = test_acc_str)

        # Set the size of the plot
        plt.gcf().set_size_inches(10, 7)

        # Set the title and other aspect of the plot
        plt.title('Whole Dataset Training Accuracy and Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Save the plot
        accuracy_train_plt.savefig('final_accuracy.png')

        # Print the final loss and accuracy
        print("Final Training Loss: ", loss[-1])
        print("Final Training Accuracy: ", accuracy[-1])
        print("Final Test Loss: ", test_loss)
        print("Final Test Accuracy: ", test_accuracy)

        # Visualize the weights
        # Split the bias and weight
        weights = weights[1:]
        
        # Min Max Normalization
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

        # Visulize each image
        for i in range(weights.shape[1]):
                # Get the image
                img = im.fromarray(cm.jet(np.reshape(weights[:,i], (32,32)), bytes=True))

                # Save the image
                img.save('weight' + str(i) + '.png')

        # Return the final loss and accuracy
        return loss[-1], accuracy[-1], test_loss, test_accuracy, weights


def run_pipeline(hyperparameters):
        """
        Run the pipeline to find the best hyperparameters

        Parameters
        ----------
        hyperparameters : argparse.Namespace
                hyperparameters for the network
        
        Returns
        -------
        para : list
                list of best hyperparameters
        """

        # build the network with the given hyperparameters
        net = network.Network(hyperparameters)

        # the range of hyperparameters to search
        pipe_rate = [0.0000001, 0.0000003, 0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003, 0.001]
        pipe_batch = [1, 2, 3, 4, 6, 8, 10] if hyperparameters.model == 'logistic' else [1, 5, 15, 25, 35, 50, 65]
        pipe_norm = [data.min_max_normalize, data.z_score_normalize]

        # store the best loss
        loss = np.inf

        # set the best hyperparameters
        indicies = None

        # loop through all the combinations of hyperparameters
        for i in range(len(pipe_rate)):
                for j in range(len(pipe_batch)):
                        for k in range(len(pipe_norm)):
                                # set the hyperparameters
                                net.rate = pipe_rate[i]
                                net.batch_size = pipe_batch[j]
                                net.norm = pipe_norm[k]

                                # run cross validation
                                _, _, l, _ = net.cross_validation(ealry_stopping=True)
                                
                                # check if the loss is better
                                if l[0] < loss:
                                        loss = l[0]
                                        indicies = None
                                        indicies = [i, j, k]

        # print the best hyperparameters and accuracy
        print("Best loss is: ", loss)
        print("Best learning rate is: ", pipe_rate[indicies[0]])
        print("Best batch size is: ", pipe_batch[indicies[1]])
        print("Best normalization is: ", pipe_norm[indicies[2]])

def run_sample():
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
        batches = [f'data_batch_{i+1}' for i in range(5)]
        
        # Loop through all the batches to get the stats of data
        stats = ''

        # Iterate through the batches
        for batch in batches:

                # Load the data
                dict_ = unpickle(batch)
                data = np.array(dict_[b'data'].reshape(-1, 3, 1024).transpose(0, 2, 1))
                labels = np.array(dict_[b'labels']) 

                # Save how many data in each class
                for i in range(10):
                        stats += f'Number of {i} in {batch}: {np.sum(labels == i)}\n'


                # Concatenate the data and labels
                full_data = data if full_data is None else np.concatenate([full_data, data])
                full_labels = labels if full_labels is None else np.concatenate([full_labels, labels])
        
        # Save the stats
        with open('stats.txt', 'w') as f:
                f.write(stats)
        
        for i in range(10):
                # Get the image with label i
                img = full_data[full_labels == i]

                # Shuffle the image
                np.random.shuffle(img)

                # Get the first image
                img = img[0]
                
                # Reshape the image from 1024x3 to 32x32x3
                img = img.reshape(32, 32, 3)

                # Save the image
                im.fromarray(img).save('img' + str(i) + '.png')

# Main function

# Parse the arguments
parser = argparse.ArgumentParser(description = 'CSE151B PA1')

# List of arguments
parser.add_argument('--batch-size', dest = 'batch_size', type = int, default = 1,
        help = 'input batch size for training (default: 1)')
parser.add_argument('--epochs', type = int, default = 100,
        help = 'number of epochs to train (default: 100)')
parser.add_argument('--learning-rate', dest = 'rate', type = float, default = 0.000001,
        help = 'learning rate (default: 0.000001)')
parser.add_argument('--min-max', dest = 'norm', action='store_const',
        default = data.z_score_normalize, const = data.min_max_normalize,
        help = 'use min max normalization on the dataset, default is z score normalization')
parser.add_argument('--k-folds', dest='k', type = int, default = 10,
        help = 'number of folds for cross-validation')
parser.add_argument('--model', type = str, default = 'logistic',
        help = 'model to use, either logistic or neural_net')
parser.add_argument('--index1', type = int, default = 0,
        help = 'index of first class to use')
parser.add_argument('--index2', type = int, default = 5,
        help = 'index of second class to use')
parser.add_argument('--function', type = str, default='run_single',
        help = 'function to run')
args = parser.parse_args()

# Save the arguments
hyperparameters = parser.parse_args()

# Decide what function to run
if hyperparameters.function == 'run_single':
        # Run the single run
        run_single(hyperparameters)

# Run the pipeline
elif hyperparameters.function == 'run_pipeline':
        # Get best hyperparameters
        run_pipeline(hyperparameters)

# Run the random sample
elif hyperparameters.function == 'run_sample':
        # Draw one random graph from each class and save it
        run_sample()
