---
title: Experiment in Logistic and Softmax Regression to Classify Different Classes of Images
author: Dongze Li, Xiaoyan He
Date: October 2022
---

# Experiment in Logistic and Softmax Regression to Classify Different Classes of Images

Source code from CSE 151B Fall 2022 PA1

## Description

We use `NumPy` and basic Python packages to build logistic and softmax models to train on the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset, which contains 10 classes of 32x32 pixel-size images. Also, we use `Matplotlib` to visulize both loss and accuracy in cross validation as well as testing. For logistic regression experiment, we used our logistic regression model to classify the images of airplane and dog as well as classify the images of cat and dog. For the first task, our
model achieved the test accuracy of 71.2%. For the second task, the model make the prediction with test accuracy of 60%. More than that, we used softmax regression to classify all the images in the dataset. It reaches the test accuracy of 31.8% for our best performed model. Also, we found that the performance of the model is related to the similarity between the class of images. We also observed that the different combinations of hyperparameters have some impact of our model’s performance.

## Getting Started

### Dependencies

* `Python3`
* `NumPy`
* `Matplotlib`
* `argparse`
* `pickle`
* `os`
* `PIL`
* `tqdm`

### Installs

* `get_data.sh` (download the dataset)
    * Simply run the bash script, then the dataset will be downloaded.

### Files

* `data.py` this is given in the starter code, and we implement different methods such as normalization, data shuffling, minibatches, and other add-ons.
* `get_data.sh` this is given in the starter code, and it is a bash script to get data from the CIFAR-10 official website.
* `image.py` this is given in the starter code, and no modification is made by us, used for save images.
* `main.py` this is given in the starter code, and it acts as the main function for this project. We have modified it to run different models with different hyperparameters. It also has different add-on functions to visulize weights, tune variables, save images, and give visulizations. By running this program with give parameters, you can see different result and get different saved files. The instruction will be specified below.
* `network.py` this is given in the starter code, and we implement the given functions as well as some other helper methods such as cross validation, accuracy calculation, etc. The neural network for both logistic and softmax regression is implemented inside this file.

### Executing Program

* Go to the correct directory where all files located
* In the terminal, run `python main.py` with or without the following argument phrases:
    * `[-h]`
    * `[--batch-size BATCH_SIZE]`
    * `[--epochs EPOCHS]`
    * `[--learning-rate RATE]`
    * `[--min-max]`
    * `[--k-folds K]`
    * `[--model MODEL]`
    * `[--index1 INDEX1]`
    * `[--index2 INDEX2]`
    * `[--function FUNCTION]`

* Here are the explanations of different argument phrases:
    * `-h`, `--help`
        * Show help messages

    * `--batch-size` + `BATCH_SIZE`   
        * Batch size for training
        * (default: `BATCH_SIZE = 1`, options: any positive `int`)

    * `--epochs` + `EPOCHS`
        * Number of epochs to train for
        * (default: `EPOCHS = 100`, options: any positive `int`)

    * `--learning-rate` + `RATE`
        * Learning rate for training
        * (default: `RATE = 0.000001`, options: any positive `float`)

    * `--min-max`
        * Use min-max normalization
        * (default: `z-score` normalization, options: use `--min-max` to switch to `min-max` normalization)
    
    * `--k-folds` + `K`
        * Number of folds for k-fold cross validation
        * (default: `K = 10`, options: any `int` larger than 1)

    * `--model` + `MODEL`
        * Model to use for training
        * (default: `MODEL = 'logistic'`, options: `'logistic'` and `'softmax'`)

    * `--index1` + `INDEX1`
        * Index of first feature to use when model is logistic
        * (default: `INDEX1 = 0`, options: any `int` between 0 and 9)

    * `--index2` + `INDEX2`
        * Index of second feature to use when model is logistic
        * (default: `INDEX2 = 5`, options: any `int` between 0 and 9, besides `INDEX1`)

    * `--function` + `FUNCTION`
        * Function to use for training
        * (default: `FUNCTION = 'run_single'`, options: `'run_single'`, `'run_pipeline'`, and `'run_sample'`)
            * `'run_single'`: run a cross validation on the training set with given hyperpatameters, then train on the whole training set and print out the test result. Loss and accuracy curve for both cross validation mean and whole trainig will be shown and saved. Visulization of weight matrix will also be saved. No early stopping in this function.
            * `'run_pipeline'`: run a pipeline of training with pre-set parameters, and user given `MODEL`. Return the best hyper parameter for the given model. Early stopping is implemented in this function.
            * `'run_sample'`: randomly draw 1 data from each class (10 total) and save the images.

Proper result will be shown as well as images will be saved at the current path if proper parameter is given and correct function is chosen.

## Help
Make sure to download the dataset first before running anything. Also, please save images in current path before running another round of `main.py` wihch will cause the old image being replaced by the new ones.

## Authors
Contributor names and contacts info (alphabetic order):

* Li, Dongze
    * dol005@ucsd.edu
* He, Xiaoyan
    * x6he@ucsd.edu

## Acknowledgments

We appreciate the help from the coruse website, Piazza, as well as TAs and Tutors' office hours. We also appreciate Professor [Garrison W. Cottrell](https://cseweb.ucsd.edu/~gary/) for his lectures and teachings.
