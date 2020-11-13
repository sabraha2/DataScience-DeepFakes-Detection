# Libraries to use 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from matplotlib import pyplot
import numpy as np
import pandas as pd
import cv2
import os
import glob
import warnings

import random
from natsort import natsorted
import json
import args

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def dataset_loader(split):
    feats_path = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/rPPG/dataset_arrays/%s_features.npy' % split
    labels_path = '/afs/crc.nd.edu/group/cvrl/scratch_32/DeepFakes/rPPG/dataset_arrays/%s_labels.npy' % split
    feats = np.load(feats_path)
    labels = np.load(labels_path)
    return feats, labels


def main():
    warnings.filterwarnings('ignore')

    arg_obj = args.get_input()
    args.print_args(arg_obj)
    crc_task_number = int(arg_obj.number) - 1

    ## Set seed to replicate experiments
    seed = 172
    np.random.seed(seed)

    train_features, train_labels = dataset_loader('train')
    print(train_features.shape, train_labels.shape)
    validation_features, validation_labels = dataset_loader('validation')
    print(validation_features.shape, validation_labels.shape)
    test_features, test_labels = dataset_loader('test')
    print(test_features.shape, test_labels.shape)

    # Parameters that can be changed 
    # ------------------------------
    # argument needed for Random Forests Model 
    trees = 2

    # for splitting the dataset, designate a percent to assign to test
    # In our case let's try 10 percent 
    test_size = 0.10

    # For Logistic Regression 
    # seed of pseudo random number generator to use when shuffling the data
    seed = 1

    # Create the machine learning models 
    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))

    ## Select model based on task array index
    models = [models[crc_task_number]]

    all_features = np.vstack((train_features, validation_features, test_features))
    all_labels = np.hstack((train_labels, validation_labels, test_labels))
    print(all_features.shape, all_labels.shape)

    for name, model in models:
        print('Cross validating: ', name)
        model_preds = cross_val_predict(model, all_features, all_labels, cv=5)
        np.save('preds/%s_kfold_subset_preds.npy' % name, model_preds)


if __name__ == "__main__":
    main()
