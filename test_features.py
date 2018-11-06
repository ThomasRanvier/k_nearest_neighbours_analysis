from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import *
from sklearn.ensemble import ExtraTreesClassifier

import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=int, help="The dataset to use. 1: Iris flower. 2: Breast cancer.")

args = parser.parse_args()

if __name__ == '__main__':

    # Choose features to train on

    if (args.dataset == 1):
        dataset = load_iris()
        features = [i for i in range(0, 4)]
    else:
        dataset = load_breast_cancer()
        features = [i for i in range(0, 30)]

    forest = ExtraTreesClassifier(n_estimators=250)

    forest.fit(dataset.data[:,features], dataset.target)
    importances = forest.feature_importances_
    
    xaxis = [x for x in range(len(features))]

    plt.title("Importance of each feature")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.bar(xaxis, importances)

    plt.show()
