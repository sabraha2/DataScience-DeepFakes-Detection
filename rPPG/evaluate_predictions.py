# Libraries to use 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, average_precision_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
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

    all_features = np.vstack((train_features, validation_features, test_features))
    all_labels = np.hstack((train_labels, validation_labels, test_labels))
    print(all_labels.shape)

    # Create the machine learning models 
    models = []
    models.append('LR')
    models.append('LDA')
    models.append('KNN')
    models.append('CART')
    models.append('RF')
    models.append('NB')
    models.append('SVM')

    for model in models:
        print(model)
        preds = np.load('preds/%s_kfold_preds.npy' % model)
        print(classification_report(all_labels, preds, digits=4))
        ave_prec = average_precision_score(all_labels, preds)
        acc = (preds == all_labels).sum() / len(preds)
        tp = ((preds == 1) * (preds == all_labels)).sum()
        fp = ((preds == 1) * (preds != all_labels)).sum()
        tn = ((preds == 0) * (preds == all_labels)).sum()
        fn = ((preds == 0) * (preds != all_labels)).sum()
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * ((prec * rec) / (prec + rec))
        spec = tn / (tn + fp)
        npv = tn / (tn + fn)
        print('Acc: ', acc)
        print('Prec: ', prec)
        print('Rec: ', rec)
        print('F1: ', f1)
        print('Spec: ', spec)
        print('NPV: ', npv)
        print('Ave Prec: ', ave_prec)
        print('Sum preds: ', np.sum(preds))
        print('Sum labels: ', np.sum(all_labels))
        print()




if __name__ == "__main__":
    main()
