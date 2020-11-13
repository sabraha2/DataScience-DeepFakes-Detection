#!/usr/bin/env python
# coding: utf-8
import os
import glob
import argparse
import contextlib
from pprint import pprint

from collections import OrderedDict

import numpy as np
import mahotas
import cv2
import h5py
from matplotlib import pyplot

import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import confusion_matrix, accuracy_score, \
    classification_report, precision_score, recall_score, f1_score, \
    roc_auc_score, average_precision_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.base import clone

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

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

# number of jobs - parallelization (where applicable)
n_jobs = -1  # -1 uses all available processors

# Define data paths
train = "train_test_data/images"  # this is dense flow atm...
# train = "train_test_data_SPARSE_FLOW/images"

# Access features and labels
data_file = 'output/data.h5'
labels_file = 'output/labels.h5'
image_ids_file = 'output/image_ids.h5'

# Define our scoring metric
scoring = "accuracy"
size = (500, 500)

# Create a directory named 'output' to hold labels and feature data
if not os.path.isdir('output'):
    os.mkdir('output')

# https://github.com/scikit-learn/scikit-learn/pull/18649
from collections import defaultdict
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable, check_random_state, _safe_indexing
from sklearn.utils import _approximate_mode
from sklearn.utils.validation import _num_samples, column_or_1d
from sklearn.utils.validation import check_array
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import type_of_target


class StratifiedGroupKFold(_BaseKFold):
    """Stratified K-Folds iterator variant with non-overlapping groups.
    This cross-validation object is a variation of StratifiedKFold that returns
    stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    The difference between GroupKFold and StratifiedGroupKFold is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class as much as possible given the
    constraint of non-overlapping groups between splits.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
        This implementation can only shuffle groups that have approximately the
        same y distribution, no global shuffle will be performed.
    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> cv = StratifiedGroupKFold(n_splits=3)
    >>> for train_idxs, test_idxs in cv.split(X, y, groups):
    ...     print("TRAIN:", groups[train_idxs])
    ...     print("      ", y[train_idxs])
    ...     print(" TEST:", groups[test_idxs])
    ...     print("      ", y[test_idxs])
    TRAIN: [1 1 2 2 4 5 5 5 5 8 8]
           [0 0 1 1 1 0 0 0 0 0 0]
     TEST: [3 3 3 6 6 7]
           [1 1 1 0 0 0]
    TRAIN: [3 3 3 4 5 5 5 5 6 6 7]
           [1 1 1 1 0 0 0 0 0 0 0]
     TEST: [1 1 2 2 8 8]
           [0 0 1 1 0 0]
    TRAIN: [1 1 2 2 3 3 3 6 6 7 8 8]
           [0 0 1 1 1 1 1 0 0 0 0 0]
     TEST: [4 5 5 5 5]
           [1 0 0 0 0]
    Notes
    -----
    The implementation is designed to:
    * Mimic the behavior of StratifiedKFold as much as possible for trivial
      groups (e.g. when each group contain only one sample).
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Stratify based on samples as much as possible while keeping
      non-overlapping groups constraint. That means that in some cases when
      there is a small number of groups containing a large number of samples
      the stratification will not be possible and the behavior will be close
      to GroupKFold.
    * Implementation is based on this kaggle kernel:
      https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
      Changelist:
      - Refactored function to a class following scikit-learn KFold interface.
      - Added heuristic for assigning group to the least populated fold in
        cases when all other criteria are equal
      - Swtch from using python ``Counter`` to ``np.unique`` to get class
        distribution
      - Added scikit-learn checks for input: checking that target is binary or
        multiclass, checking passed random state, checking that number of
        splits is less than number of members in each class, checking that
        least populated class has more members than there are splits.
    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        n_smallest_class = np.min(y_cnt)
        if self.n_splits > n_smallest_class:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is less than n_splits=%d."
                           % (n_smallest_class, self.n_splits)), UserWarning)
        labels_num = len(y_cnt)
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        for label, group in zip(y_inv, groups):
            y_counts_per_group[group][label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        groups_and_y_counts = list(y_counts_per_group.items())
        if self.shuffle:
            rng.shuffle(groups_and_y_counts)

        for group, y_counts in sorted(groups_and_y_counts,
                                      key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = np.inf
            min_samples_in_fold = np.inf
            for i in range(self.n_splits):
                y_counts_per_fold[i] += y_counts
                std_per_label = []
                for label in range(labels_num):
                    std_per_label.append(np.std(
                        [y_counts_per_fold[j][label] / y_cnt[label]
                         for j in range(self.n_splits)]))
                y_counts_per_fold[i] -= y_counts
                fold_eval = np.mean(std_per_label)
                samples_in_fold = np.sum(y_counts_per_fold[i])
                is_current_fold_better = (
                        np.isclose(fold_eval, min_eval)
                        and samples_in_fold < min_samples_in_fold
                        or fold_eval < min_eval
                )
                if is_current_fold_better:
                    min_eval = fold_eval
                    min_samples_in_fold = samples_in_fold
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(group)

        for i in range(self.n_splits):
            test_indices = [idx for idx, group in enumerate(groups)
                            if group in groups_per_fold[i]]
            yield test_indices


if not (os.path.isfile(data_file) and os.path.isfile(labels_file) and
        os.path.isfile(image_ids_file)):
    print('Generating data, hang in there...')
    # Setting number of bins for color histogram
    bins = 8


    # https://stackoverflow.com/a/58936697/6557588
    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""

        class TqdmBatchCompletionCallback(
            joblib.parallel.BatchCompletionCallBack):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()


    # Extract Features
    # Extracting color feature descriptors
    def color_features(image, mask=None):
        # Need to convert image to HSV color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Calculate the color histogram and normalize
        histogram = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins],
                                 [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram)
        return histogram.flatten()


    # Extracting shape features
    def shape_features(image):
        # Image need to be in grayscale first
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape_features = cv2.HuMoments(cv2.moments(image)).flatten()
        return shape_features


    # Extracting texture feature descriptors
    def texture_features(image):
        # Image needs to be in grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # calculate the texture feature vector
        texture = mahotas.features.haralick(gray_img).mean(axis=0)
        return texture


    def preprocess_image(image_path):
        # Read in the image
        image = cv2.imread(image_path)
        # Resize the image
        image = cv2.resize(image, size)
        # Extract features from each image
        color = color_features(image)
        shape = shape_features(image)
        texture = texture_features(image)
        # Concatenate features
        global_feature = np.hstack([color, shape, texture])
        return global_feature


    training_labels = os.listdir(train)
    training_labels.sort()
    print('training_labels', training_labels)

    global_features = []
    labels = []
    image_ids = []

    # Access subfolders in train
    for name in training_labels:
        # Get the subfolder name which is the label for that set of images
        sub_folder = os.path.join(train, name)
        # Loop over the images in the sub-folder
        list_of_images = os.listdir(sub_folder)
        # Add labels & image ids
        labels.extend([name] * len(list_of_images))
        image_ids.extend([img.split('_', 1)[0] for img in list_of_images])

        # Parallel preprocess time
        with tqdm_joblib(tqdm(desc='process {}'.format(name),
                              total=len(list_of_images))):
            global_features.extend(Parallel(n_jobs=-1)(
                delayed(preprocess_image)(os.path.join(sub_folder, image))
                for image in list_of_images
            ))
        print("Finished processing sub-folder: {}".format(name))

    print("Global feature extraction is complete :D")

    # Encode the labels to unique name
    target_names = np.unique(labels)
    target = LabelEncoder().fit_transform(labels)

    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled = scaler.fit_transform(global_features)

    # Save features and labels in HDf5 format
    data_features = h5py.File(data_file, 'w')
    data_features.create_dataset('dataset', data=np.array(rescaled))

    label = h5py.File(labels_file, 'w')
    label.create_dataset('dataset', data=np.array(target))

    image_ids_f = h5py.File(image_ids_file, 'w')
    image_ids_ascii = [s.encode('ascii') for s in image_ids]
    image_ids_f.create_dataset('dataset', data=image_ids_ascii)

    data_features.close()
    label.close()
    image_ids_f.close()
else:
    print('Using existing data and labels found in the output dir')

# Get the training labels
train_labels = os.listdir(train)
train_labels.sort()

# Create the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed, n_jobs=n_jobs)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_jobs=n_jobs)))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF',
               RandomForestClassifier(n_estimators=trees, random_state=seed,
                                      n_jobs=n_jobs)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

# Initialize empty variabes to hold results and names
results = []
names = []

# Import our global feature vectors and their labels
print('Reading in data...')
data = h5py.File(data_file, 'r')
label = h5py.File(labels_file, 'r')
image_ids_f = h5py.File(image_ids_file, 'r')

global_features_string = data['dataset']
global_labels_string = label['dataset']
global_image_ids_string = image_ids_f['dataset']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)
global_image_ids = np.array([s.decode('utf-8')
                             for s in global_image_ids_string])

data.close()
label.close()
image_ids_f.close()

if args.debug:
    print('Debug mode! Using less data to test things out...')
    idx_0 = np.where(global_labels == 0)[0][:100]
    idx_1 = np.where(global_labels == 1)[0][:100]
    idx = np.r_[idx_0, idx_1]
    global_features = global_features[idx]
    global_labels = global_labels[idx]
    global_image_ids = global_image_ids[idx]

# Verifying
print("Feature vector shape - {}".format(global_features.shape))
print("Feature vector shape - {}".format(global_labels.shape))

# Splitting into train and test data
# create mapping of unique image_id to indices
global_images = {}
for i, image_id in enumerate(global_image_ids):
    image_id_idxs = global_images.get(image_id, [])
    image_id_idxs.append(i)
    global_images[image_id] = image_id_idxs

global_images = OrderedDict(
    sorted(global_images.items(), key=lambda it: it[0])
)
# videowise labels
global_labels_by_id = np.asarray(
    [global_labels[idxs[0]] for idxs in global_images.values()])

# Stratified train test split
train_ids, test_ids, train_id_idxs, test_id_idxs = train_test_split(
    list(global_images.keys()),
    np.arange(len(global_images)),
    test_size=test_size,
    random_state=seed,
    stratify=global_labels_by_id
)

trainData = []
trainLabels = []
trainGroups = []
for i, train_id in enumerate(train_ids):
    train_idxs = global_images[train_id]
    trainGroups.extend([i] * len(train_idxs))
    for train_idx in train_idxs:
        trainData.append(global_features[train_idx])
        trainLabels.append(global_labels[train_idx])
trainData = np.vstack(trainData)
trainLabels = np.asarray(trainLabels)
trainGroups = np.asarray(trainGroups)

testData = []
testLabels = []
testGroups = []
for i, test_id in enumerate(test_ids):
    test_idxs = global_images[test_id]
    testGroups.extend([i] * len(test_idxs))
    for test_idx in test_idxs:
        testData.append(global_features[test_idx])
        testLabels.append(global_labels[test_idx])
testData = np.vstack(testData)
testLabels = np.asarray(testLabels)
testGroups = np.asarray(testGroups)


def class_counts(class_array):
    return dict(zip(*np.unique(class_array, return_counts=True)))


# Verify values correctly correspond
train_video_labels = global_labels_by_id[train_id_idxs]
test_video_labels = global_labels_by_id[test_id_idxs]
print('Train videos - {} ({})'.format(len(train_ids),
                                      class_counts(train_video_labels)))
print("Train data - {} ({})".format(trainData.shape,
                                    class_counts(trainLabels)))
print("Train labels - {}".format(trainLabels.shape))
print('Test videos - {} ({})'.format(len(test_ids),
                                     class_counts(test_video_labels)))
print("Test data - {} ({})".format(testData.shape, class_counts(testLabels)))
print("Test labels - {}".format(testLabels.shape))

print()
header = 'model | mean score | std score'
print(len(header) * '-')
print(header)
print(len(header) * '-')


def videowise_accuracy(y_true, y_pred, groups, positive_thresh=0.5):
    assert len(y_true) == len(y_pred) == len(groups)

    score = 0
    group_set = set(groups)
    for group in group_set:
        idxs = np.where(groups == group)
        y_pred_i = y_pred[idxs]
        y_true_i = y_true[idxs]
        assert np.all(y_true_i[0] == y_true_i), 'bad grouping'

        y_true_i = y_true_i[0]
        y_pred_i = int((np.sum(y_pred_i == 1) / len(idxs)) >= positive_thresh)
        score += int(y_pred_i == y_true_i)
    return score / len(group_set)


def videowise_preds(y_true, y_pred, groups):
    assert len(y_true) == len(y_pred) == len(groups)

    score = 0
    group_set = set(groups)
    y_trues = []
    y_preds = []
    for group in group_set:
        idxs = np.where(groups == group)
        y_pred_i = y_pred[idxs]
        y_true_i = y_true[idxs]
        assert np.all(y_true_i[0] == y_true_i), 'bad grouping'

        y_true_i = y_true_i[0]
        # Proportion of class 1 predicted
        y_pred_i = np.sum(y_pred_i == 1) / len(y_pred_i)

        y_trues.append(y_true_i)
        y_preds.append(y_pred_i)
    return np.asarray(y_trues), np.asarray(y_preds)


def metrics_pls(y_true, y_pred, threshold=0.5):
    y_pred_onehot = (y_pred >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred_onehot),
        'f1': f1_score(y_true, y_pred_onehot),
        'auc': roc_auc_score(y_true, y_pred),
        'avg_precision': average_precision_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred_onehot),
        'recall': recall_score(y_true, y_pred_onehot),
    }


for name, model in models:
    print()
    print('Beginning {}...'.format(name))
    # kfold = KFold(n_splits=5, random_state=seed, shuffle=True)
    # kfold = GroupKFold(n_splits=5)
    kfold = StratifiedGroupKFold(n_splits=5)
    # cv_results = cross_val_score(model, trainData, trainLabels, cv=kfold, scoring=scoring)

    cv_results = []
    iamthebestthresholds = []
    for train_index, test_index in kfold.split(trainData,
                                               trainLabels,
                                               trainGroups):
        X_train, X_test = trainData[train_index], trainData[test_index]
        y_train, y_test = trainLabels[train_index], trainLabels[test_index]
        groups_train, groups_test = trainGroups[train_index], trainGroups[
            test_index]

        # Print split stats
        print('Split stats:')
        print('Train videos - {} ({})'.format(len(set(groups_train)),
                                              class_counts(train_video_labels[
                                                               groups_train])))
        print("Train data - {} ({})".format(X_train.shape,
                                            class_counts(y_train)))
        print("Train labels - {}".format(y_train.shape))
        print('Val videos - {} ({})'.format(len(set(groups_test)),
                                            class_counts(train_video_labels[
                                                             groups_test])))
        print("Val data - {} ({})".format(X_test.shape, class_counts(y_test)))
        print("Val labels - {}".format(y_test.shape))

        model_cv = clone(model)
        model_cv.fit(X_train, y_train)

        y_pred = model_cv.predict(X_test)
        # acc = videowise_accuracy(y_test, y_pred, groups_test)
        y_test_vw, y_pred_vw = videowise_preds(y_test, y_pred, groups_test)
        # NAIVE THRESHOLD TIME
        best_metrics = None
        best_threshold = None
        for threshold in np.linspace(.05, .95, 19):
            metrics = metrics_pls(y_test_vw, y_pred_vw, threshold)
            if best_threshold is None or best_metrics['f1'] < metrics['f1']:
                best_threshold = threshold
                best_metrics = metrics
        print('split scores (threshold {}):'.format(best_threshold))
        pprint(best_metrics)
        cv_results.append(best_metrics)
        iamthebestthresholds.append(best_threshold)
    iamthebestthreshold = np.median(iamthebestthresholds)
    print('iamthebestthreshold', iamthebestthreshold)

    # cv_results = np.asarray(cv_results)
    results.append(cv_results)
    names.append(name)

    agg_metrics = {}
    for result in cv_results:
        for metric_name, metric_val in result.items():
            aggm = agg_metrics.get(metric_name, [])
            aggm.append(metric_val)
            agg_metrics[metric_name] = aggm
    for metric_name, metric_vals in agg_metrics.items():
        msg = "%s %s: %f (%f)" % (name, metric_name, np.mean(metric_vals),
                                  np.std(metric_vals))
        print(msg)

    # Evaluate test accuracy
    model_test = clone(model)
    model_test.fit(trainData, trainLabels)
    test_preds = model_test.predict(testData)

    # test_acc = videowise_accuracy(testLabels, test_preds, testGroups)
    # Use best-found threshold from train data on test
    y_test_vw, y_pred_vw = videowise_preds(testLabels, test_preds,
                                           testGroups)
    metrics = metrics_pls(y_test_vw, y_pred_vw, iamthebestthreshold)
    print('Test scores:')
    pprint(metrics)

    print()
