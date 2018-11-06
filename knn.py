from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import *

import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=int, help="The dataset to use. 1: Iris flower. 2: Breast cancer.")
parser.add_argument("--percent", type=float, default=0.33, help="The percentage of the dataset to be used as a test. Default 0.33")

args = parser.parse_args()


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

    if int(args.dataset) == 1:
        x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
        y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig = plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train,s=20, edgecolor='k')
        plt.title("iris, 0.33, k: %d" % k)
        plt.xlabel('Feature 2')
        plt.ylabel('Feature 3')
        fig.savefig('images/1-033-' + str(k) + '.png', bbox_inches='tight')

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

def load_dataset(dataset, features, test_size):
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
    # Load the dataset
    
    print('You are using the features:')
    for x in features:
        print x,"-", dataset.feature_names[x]
    
    X = dataset.data[:,features]
    Y = dataset.target
    
    # Split the dataset into a training and a test set
    return train_test_split(X, Y, test_size=test_size, random_state=int(time.time()))


if __name__ == '__main__':
    # Choose features to train on

    # The maximum value of k
    k_max = 30

    if int(args.dataset) == 1:
        dataset = load_iris()
        features = [2, 3]
    elif int(args.dataset) == 2:
        dataset = load_breast_cancer()
        features = [0, 2, 3, 6, 7, 20, 22, 23, 26, 27]

    # Load the dataset with a test/training set ratio of 0.33
    x_train, x_test, y_train, y_test = load_dataset(dataset, features, args.percent)
    
    # Lists to save results in
    train_scores = []
    test_scores = []
    
    # Train the classifier with different values for k and save the accuracy 
    # achieved on the training and test sets
    for k in range(1, k_max):
        train_score, test_score = evaluate_knn(train_knn(x_train, y_train, k), x_train, y_train, x_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    print "-----------------"
    print "The best test score is reached for ", np.argmax(test_scores), " neighbours, with a score of ", np.amax(test_scores) 
    print "The test score mean is ", np.mean(test_scores)

    # Construct plot
    plt.title('KNN results')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Accuracy')
    
    # Create x-axis
    xaxis = [x for x in range(1, k_max)]
    
    # Plot the test and training scores with labels
    fig = plt.figure(figsize=(8, 6))
    plt.plot(xaxis, train_scores, label='Training score')
    plt.plot(xaxis, test_scores, label='Test score')
    
    # Show the figure
    plt.legend()
    fig.savefig('images/1-033.png', bbox_inches='tight')
