from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import *

import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=int, help="The dataset to use. 1: Iris flower. 2: Breast cancer.")
parser.add_argument("--percent", type=float, default=0.33, help="The percentage of the dataset to be used as a test. Default 0.33")
parser.add_argument("--iterations", type=int, default=500, help="Number of iterations to test. Default 500.")

args = parser.parse_args()

def func(x, a, b, c):
    return a * np.exp(-(x-b)**2/(2*c**2))

def train_knn(x_train, y_train, k):
    """
    Given training data (input and output), train a k-NN classifier.

    Input:    x/y_train - Two arrays of equal length, one with input data and 
              one with the correct labels. 
              k - number of neighbors considered when training the classifier.
    Returns:  The trained classifier
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    return knn

def evaluate_knn(knn, x_train, y_train, x_test, y_test):
    """
    Given a trained classifier, its training data, and test data, calculate
    the accuracy on the training and test sets.
    
    Input:    knn - A trained k-nn classifier
              x/y_train - Training data
              x/y_test  - Test data
    
    Returns:  A tuple (train_acc, test_acc) with the resulting accuracies,
              obtained when using the classifier on the given data.
    """
    train_score = knn.score(x_train, y_train)
    test_score = knn.score(x_test, y_test)
    return (train_score, test_score)

def load_dataset(dataset, features, test_size, random):
    """
    Loads the iris or breast cancer datasets with the given features and 
    train/test ratio.
    
    Input:    name - Either "iris" or "breastcancer"
              features - An array with the indicies of the features to load
              test_size - How large part of the dataset to be used as test data.
                          0.33 would give a test set 33% of the total size.
    Returns:  Arrays x_train, x_test, y_train, y_test that correspond to the
              training/test sets.
    """
    X = dataset.data[:,features]
    Y = dataset.target
    
    return train_test_split(X, Y, test_size=test_size, random_state=random)


if __name__ == '__main__':
    randomness = int(time.time())

    k_max = 30

    if int(args.dataset) == 1:
        dataset = load_iris()
        features = [2, 3]
    elif int(args.dataset) == 2:
        dataset = load_breast_cancer()
        features = [0, 2, 3, 6, 7, 20, 22, 23, 26, 27]

    ks_max = []

    for i in range(args.iterations):
        if i % 50 == 0:
            print "Iteration ", i, " reached"

        x_train, x_test, y_train, y_test = load_dataset(dataset, features, args.percent, randomness)

        train_scores = []
        test_scores = []

        for k in range(1, k_max):
            train_score, test_score = evaluate_knn(train_knn(x_train, y_train, k), x_train, y_train, x_test, y_test)
            train_scores.append(train_score)
            test_scores.append(test_score)

        for v in [index+1 for index, i in enumerate(test_scores) if i == np.amax(test_scores)]:
            ks_max.append(v)

        randomness = randomness + 1
    
    print "-----------------"
    print np.bincount(ks_max)

    print "-----------------"
    print "The best scores has been reached for ", np.bincount(ks_max).argmax(), " neighbours."

    plt.title('KNN results')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Occuerences')
    
    occurences = np.bincount(ks_max)

    xaxis = [x for x in range(len(occurences))]

    popt, pcov = curve_fit(func, xaxis, occurences)
    gaussian = func(xaxis, *popt)
    print "K with the highest point on gaussian: ", np.argmax(gaussian)

    z_3 = np.polyfit(xaxis, occurences, 3)
    f_3 = np.poly1d(z_3)
    z_5 = np.polyfit(xaxis, occurences, 5)
    f_5 = np.poly1d(z_5)

    poly_3 = f_3(xaxis)
    print "K with the highest point on polynom degree 3: ", np.argmax(poly_3)
    poly_5 = f_5(xaxis)
    print "K with the highest point on polynom degree 5: ", np.argmax(poly_5)

    plt.bar(xaxis, occurences, label='occurences')
    plt.plot(xaxis, gaussian, 'r', lw=2, label='gaussian fit')
    plt.plot(xaxis, poly_3, 'g', lw=2, label='polynomial degree 3 fit')
    plt.plot(xaxis, poly_5, 'y', lw=2, label='polynomial degree 5 fit')
    
    plt.legend()
    plt.show()
