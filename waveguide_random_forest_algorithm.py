""" waveguide_random_forest_algorithm

Use the Random Forest Machine Learning algorithm to find the mode information from an x-y cross-section.

Author: Jennifer Houle
Date: 4/22/2022

"""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

def random_forest_machine_learning(datafile="waveguide_data.npz"):
    """
    Run a single random forest classifier on Ex, Ey, and Ez data to find the mode.
    :param datafile: this defaults to the name saved in final_project_main.py, but can be specified to other saved data
    :return: Cross-validation scores for each m, n, TE/TM; error rates on test data set for m, n, TE/TM, any incorrect.
    """
    data = np.load(datafile)
    lst = data.files
    output_data = data[lst[0]] # This is the labels (m, n, freq, TE[=1]/TM[=0])
    input_data = data[lst[1]] # This is the
                              # [Ex mag, Ex phase, Ey mag, Ey phase, Ez mag, Ez phase,
                              #  Hx mag, Ex phase, Hy mag, Hy phase, Hz mag, Hz phase]
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(output_data, output_data[:,2]):
        train_set_X = input_data[train_index]
        strat_train_set_Y = output_data[train_index]
        test_set_X = input_data[test_index]
        strat_test_set_Y = output_data[test_index]
    Y_train_class = np.copy(strat_train_set_Y.astype('int32'))
    Y_test_class = np.copy(strat_test_set_Y.astype('int32'))

    forest_class = RandomForestClassifier()


    ex_magnitude_train = np.copy(train_set_X[:, 0, :, :])
    ex_magnitude_train = np.reshape(ex_magnitude_train, (ex_magnitude_train.shape[0], ex_magnitude_train.shape[1]*ex_magnitude_train.shape[2]))
    ex_magnitude_train_normalized = preprocessing.normalize(ex_magnitude_train, norm='l2')
    ey_magnitude_train = np.copy(train_set_X[:, 2, :, :])
    ey_magnitude_train = np.reshape(ey_magnitude_train, (ey_magnitude_train.shape[0], ey_magnitude_train.shape[1]*ey_magnitude_train.shape[2]))
    ey_magnitude_train_normalized = preprocessing.normalize(ey_magnitude_train, norm='l2')
    ez_magnitude_train = np.copy(train_set_X[:, 4, :, :])
    ez_magnitude_train = np.reshape(ez_magnitude_train, (ez_magnitude_train.shape[0], ez_magnitude_train.shape[1]*ez_magnitude_train.shape[2]))
    ez_magnitude_train_normalized = preprocessing.normalize(ez_magnitude_train, norm='l2')


    m_train = Y_train_class[:, 0]
    n_train = Y_train_class[:, 1]
    modes_train = Y_train_class[:, 2]


    ### Single ML model trains on the Ex, Ey, and Ez data, since that is all that is necessary
    ex_ey_magnitude_train_normalized = np.concatenate((ex_magnitude_train_normalized, ey_magnitude_train_normalized), axis=1)
    ex_ey_ez_magnitude_train_normalized = np.concatenate((ex_ey_magnitude_train_normalized, ez_magnitude_train_normalized), axis=1)
    y_multilabel = np.c_[m_train, n_train, modes_train]

    y_train_tree_pred = cross_val_predict(forest_class, ex_ey_ez_magnitude_train_normalized, y_multilabel, cv=3)

    print("Random Forest Algorithm")
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro"))

    # Look at test set:
    forest_class.fit(ex_ey_ez_magnitude_train_normalized, y_multilabel)

    ex_magnitude_test = np.copy(test_set_X[:, 0, :, :])
    ex_magnitude_test = np.reshape(ex_magnitude_test, (
    ex_magnitude_test.shape[0], ex_magnitude_test.shape[1] * ex_magnitude_test.shape[2]))
    ex_magnitude_test_normalized = preprocessing.normalize(ex_magnitude_test, norm='l2')
    ey_magnitude_test = np.copy(test_set_X[:, 2, :, :])
    ey_magnitude_test = np.reshape(ex_magnitude_test, (
    ey_magnitude_test.shape[0], ey_magnitude_test.shape[1] * ey_magnitude_test.shape[2]))
    ey_magnitude_test_normalized = preprocessing.normalize(ey_magnitude_test, norm='l2')
    ez_magnitude_test = np.copy(test_set_X[:, 4, :, :])
    ez_magnitude_test = np.reshape(ez_magnitude_test, (
    ez_magnitude_test.shape[0], ez_magnitude_test.shape[1] * ez_magnitude_test.shape[2]))
    ez_magnitude_test_normalized = preprocessing.normalize(ez_magnitude_test, norm='l2')

    ### Single ML model trains on the Ex, Ey, and Ez data, since that is all that is necessary
    ex_ey_magnitude_test_normalized = np.concatenate((ex_magnitude_test_normalized, ey_magnitude_test_normalized), axis=1)
    ex_ey_ez_magnitude_test_normalized = np.concatenate((ex_ey_magnitude_test_normalized, ez_magnitude_test_normalized), axis=1)

    m_test = Y_test_class[:, 0]
    n_test = Y_test_class[:, 1]
    modes_test = Y_test_class[:, 2]

    y_multilabel_test = np.c_[m_test, n_test, modes_test]

    # Predict all the values based on the classifiers and put them in an array
    predicted_values = np.zeros((Y_test_class.shape[0], 3))

    # Values will be saved in predicted_values as [m, n, TM=0/TE=1]
    for trial_element in range(Y_test_class.shape[0]):
        predicted_values[trial_element, :] = forest_class.predict(
            ex_ey_ez_magnitude_test_normalized[trial_element, :].reshape(1, -1))

    test_error_rates = np.zeros(4)
    print("Test Data Set:")
    error_rate = 0
    for idx in range(Y_test_class.shape[0]):
        if (predicted_values[idx, 0] != Y_test_class[idx, 0]):
            error_rate += 1
    test_error_rates[0] = error_rate / Y_test_class.shape[0]
    print(f"\tError Rate for m = {test_error_rates[0]}")
    error_rate = 0
    for idx in range(Y_test_class.shape[0]):
        if (predicted_values[idx, 1] != Y_test_class[idx, 1]):
            error_rate += 1
    test_error_rates[1] = error_rate / Y_test_class.shape[0]
    print(f"\tError Rate for n = {test_error_rates[1]}")
    error_rate = 0
    for idx in range(Y_test_class.shape[0]):
        if (predicted_values[idx, 2] != Y_test_class[idx, 2]):
            error_rate += 1
    test_error_rates[2] = error_rate / Y_test_class.shape[0]
    print(f"\tError Rate for TE/TM = {test_error_rates[2]}")
    error_rate = 0
    for idx in range(Y_test_class.shape[0]):
        if (predicted_values[idx, 0] != Y_test_class[idx, 0]) | (predicted_values[idx, 1] != Y_test_class[idx, 1]) | (
                predicted_values[idx, 2] != Y_test_class[idx, 2]):
            error_rate += 1
    test_error_rates[3] = error_rate / Y_test_class.shape[0]
    print(f"\tError Rate for m, n, and TE/TM = {test_error_rates[3]}")

    return f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"), \
           f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"), \
           f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro"), \
           test_error_rates[0], test_error_rates[1], test_error_rates[2], test_error_rates[3]