""" compare_machine_learning_defaults.py

Compare the default options for various multiclass, multioutput ML algorithms on the
waveguide data.

Author: Jennifer Houle
Date: 4/22/2022

"""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score


def compare_machine_learning_defaults(datafile="waveguide_data.npz"):
    """
    Run the 6 multi-class, multi-output classifiers on Ex and Ez data to find the mode (m, n, TE/TM).
    :param datafile: this defaults to the name saved in final_project_main.py, but can be specified to other saved data
    :return: array of cross-validation results (m, n, TE/TM) for each of the ML classifiers (default setting except radius=5.0).
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

    tree_class = DecisionTreeClassifier()


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

    model_cross_val_scores = np.zeros((6, 3))
    ### Single ML model trains on the Ex and Ez data, since that is all that is necessary
    ex_ez_magnitude_train_normalized = np.concatenate((ex_magnitude_train_normalized, ez_magnitude_train_normalized), axis=1)
    y_multilabel = np.c_[m_train, n_train, modes_train]

    print("Train on Ex and Ez only:")
    print("neighbors.KNeighborsClassifier:")
    knn_clf = KNeighborsClassifier()
    y_train_knn_pred = cross_val_predict(knn_clf, ex_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_knn_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_knn_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_knn_pred[:, 2], average="macro"))
    model_cross_val_scores[0, 0] = f1_score(y_multilabel[:, 0], y_train_knn_pred[:, 0], average="macro")
    model_cross_val_scores[0, 1] = f1_score(y_multilabel[:, 1], y_train_knn_pred[:, 1], average="macro")
    model_cross_val_scores[0, 2] = f1_score(y_multilabel[:, 2], y_train_knn_pred[:, 2], average="macro")

    print("tree.DecisionTreeClassifier:")
    y_train_tree_pred = cross_val_predict(tree_class, ex_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro"))
    model_cross_val_scores[1, 0] = f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro")
    model_cross_val_scores[1, 1] = f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro")
    model_cross_val_scores[1, 2] = f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro")

    print("tree.ExtraTreeClassifier")
    from sklearn.tree import ExtraTreeClassifier
    tree_class = ExtraTreeClassifier()
    y_train_tree_pred = cross_val_predict(tree_class, ex_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro"))
    model_cross_val_scores[2, 0] = f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro")
    model_cross_val_scores[2, 1] = f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro")
    model_cross_val_scores[2, 2] = f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro")

    print("ensemble.ExtraTreeClassifier")
    from sklearn.ensemble import ExtraTreesClassifier
    tree_class = ExtraTreesClassifier()
    y_train_tree_pred = cross_val_predict(tree_class, ex_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro"))
    model_cross_val_scores[3, 0] = f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro")
    model_cross_val_scores[3, 1] = f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro")
    model_cross_val_scores[3, 2] = f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro")

    print("ensemble.RandomForestClassifier")
    from sklearn.ensemble import RandomForestClassifier
    forest_class = RandomForestClassifier()
    y_train_forest_pred = cross_val_predict(forest_class, ex_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_forest_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_forest_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_forest_pred[:, 2], average="macro"))
    model_cross_val_scores[4, 0] = f1_score(y_multilabel[:, 0], y_train_forest_pred[:, 0], average="macro")
    model_cross_val_scores[4, 1] = f1_score(y_multilabel[:, 1], y_train_forest_pred[:, 1], average="macro")
    model_cross_val_scores[4, 2] = f1_score(y_multilabel[:, 2], y_train_forest_pred[:, 2], average="macro")

    print("neighbors.RadiusNeighborsClassifier")
    from sklearn.neighbors import RadiusNeighborsClassifier
    radius_class = RadiusNeighborsClassifier(radius=5.0)
    y_train_radius_pred = cross_val_predict(radius_class, ex_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_radius_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_radius_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_radius_pred[:, 2], average="macro"))
    model_cross_val_scores[5, 0] = f1_score(y_multilabel[:, 0], y_train_radius_pred[:, 0], average="macro")
    model_cross_val_scores[5, 1] = f1_score(y_multilabel[:, 1], y_train_radius_pred[:, 1], average="macro")
    model_cross_val_scores[5, 2] = f1_score(y_multilabel[:, 2], y_train_radius_pred[:, 2], average="macro")
    print("\n")

    return model_cross_val_scores


def compare_machine_learning_all_e_defaults(datafile="waveguide_data.npz"):
    """
    Run the 6 multi-class, multi-output classifiers on Ex, Ey, and Ez data to find the mode (m, n, TE/TM).
    :param datafile: this defaults to the name saved in final_project_main.py, but can be specified to other saved data
    :return: array of cross-validation results (m, n, TE/TM) for each of the ML classifiers (default setting except radius=5.0).
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

    tree_class = DecisionTreeClassifier()


    ex_magnitude_train = np.copy(train_set_X[:, 0, :, :])
    ex_magnitude_train = np.reshape(ex_magnitude_train, (ex_magnitude_train.shape[0], ex_magnitude_train.shape[1]*ex_magnitude_train.shape[2]))
    ex_magnitude_train_normalized = preprocessing.normalize(ex_magnitude_train, norm='l2')
    ey_magnitude_train = np.copy(train_set_X[:, 2, :, :])
    ey_magnitude_train = np.reshape(ey_magnitude_train, (ey_magnitude_train.shape[0], ey_magnitude_train.shape[1]*ey_magnitude_train.shape[2]))
    ey_magnitude_train_normalized = preprocessing.normalize(ey_magnitude_train, norm='l2')
    ez_magnitude_train = np.copy(train_set_X[:, 4, :, :])
    ez_magnitude_train = np.reshape(ez_magnitude_train, (ez_magnitude_train.shape[0], ez_magnitude_train.shape[1]*ez_magnitude_train.shape[2]))
    ez_magnitude_train_normalized = preprocessing.normalize(ez_magnitude_train, norm='l2')


    model_cross_val_scores = np.zeros((6, 3))
    m_train = Y_train_class[:, 0]
    n_train = Y_train_class[:, 1]
    modes_train = Y_train_class[:, 2]


    ### Single ML model trains on the Ex, Ey, and Ez data, since that is all that is necessary
    ex_ey_magnitude_train_normalized = np.concatenate((ex_magnitude_train_normalized, ey_magnitude_train_normalized), axis=1)
    ex_ey_ez_magnitude_train_normalized = np.concatenate((ex_ey_magnitude_train_normalized, ez_magnitude_train_normalized), axis=1)
    y_multilabel = np.c_[m_train, n_train, modes_train]

    print("Train on Ex, Ey, and Ez")
    print("neighbors.KNeighborsClassifier:")
    knn_clf = KNeighborsClassifier()
    y_train_knn_pred = cross_val_predict(knn_clf, ex_ey_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_knn_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_knn_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_knn_pred[:, 2], average="macro"))
    model_cross_val_scores[0, 0] = f1_score(y_multilabel[:, 0], y_train_knn_pred[:, 0], average="macro")
    model_cross_val_scores[0, 1] = f1_score(y_multilabel[:, 1], y_train_knn_pred[:, 1], average="macro")
    model_cross_val_scores[0, 2] = f1_score(y_multilabel[:, 2], y_train_knn_pred[:, 2], average="macro")

    print("tree.DecisionTreeClassifier:")
    y_train_tree_pred = cross_val_predict(tree_class, ex_ey_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro"))
    model_cross_val_scores[1, 0] = f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro")
    model_cross_val_scores[1, 1] = f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro")
    model_cross_val_scores[1, 2] = f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro")

    print("tree.ExtraTreeClassifier")
    from sklearn.tree import ExtraTreeClassifier
    tree_class = ExtraTreeClassifier()
    y_train_tree_pred = cross_val_predict(tree_class, ex_ey_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro"))
    model_cross_val_scores[2, 0] = f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro")
    model_cross_val_scores[2, 1] = f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro")
    model_cross_val_scores[2, 2] = f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro")

    print("ensemble.ExtraTreeClassifier")
    from sklearn.ensemble import ExtraTreesClassifier
    tree_class = ExtraTreesClassifier()
    y_train_tree_pred = cross_val_predict(tree_class, ex_ey_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro"))
    model_cross_val_scores[3, 0] = f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro")
    model_cross_val_scores[3, 1] = f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro")
    model_cross_val_scores[3, 2] = f1_score(y_multilabel[:, 2], y_train_tree_pred[:, 2], average="macro")

    print("ensemble.RandomForestClassifier")
    from sklearn.ensemble import RandomForestClassifier
    forest_class = RandomForestClassifier()
    y_train_forest_pred = cross_val_predict(forest_class, ex_ey_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_forest_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_forest_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_forest_pred[:, 2], average="macro"))
    model_cross_val_scores[4, 0] = f1_score(y_multilabel[:, 0], y_train_forest_pred[:, 0], average="macro")
    model_cross_val_scores[4, 1] = f1_score(y_multilabel[:, 1], y_train_forest_pred[:, 1], average="macro")
    model_cross_val_scores[4, 2] = f1_score(y_multilabel[:, 2], y_train_forest_pred[:, 2], average="macro")

    print("neighbors.RadiusNeighborsClassifier")
    from sklearn.neighbors import RadiusNeighborsClassifier
    radius_class = RadiusNeighborsClassifier(radius=5.0)
    y_train_radius_pred = cross_val_predict(radius_class, ex_ey_ez_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_radius_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_radius_pred[:, 1], average="macro"))
    print("\tmode f1 score: ", f1_score(y_multilabel[:, 2], y_train_radius_pred[:, 2], average="macro"))
    model_cross_val_scores[5, 0] = f1_score(y_multilabel[:, 0], y_train_radius_pred[:, 0], average="macro")
    model_cross_val_scores[5, 1] = f1_score(y_multilabel[:, 1], y_train_radius_pred[:, 1], average="macro")
    model_cross_val_scores[5, 2] = f1_score(y_multilabel[:, 2], y_train_radius_pred[:, 2], average="macro")
    print("\n")

    return model_cross_val_scores

def compare_machine_learning_2_algorithms(datafile="waveguide_data.npz"):
    """
    Run the 6 multi-class, multi-output classifiers on Ex to find the mode's m and n
    Run SGD to find the mode's TE/TM
    :param datafile: this defaults to the name saved in final_project_main.py, but can be specified to other saved data
    :return: average score for SGD classifier for TE/TM; array of cross-validation results (m, n) for each of the ML
    classifiers (default setting except radius=5.0).
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

    tree_class = DecisionTreeClassifier()


    ex_magnitude_train = np.copy(train_set_X[:, 0, :, :])
    ex_magnitude_train = np.reshape(ex_magnitude_train, (ex_magnitude_train.shape[0], ex_magnitude_train.shape[1]*ex_magnitude_train.shape[2]))
    ex_magnitude_train_normalized = preprocessing.normalize(ex_magnitude_train, norm='l2')
    ey_magnitude_train = np.copy(train_set_X[:, 2, :, :])
    ey_magnitude_train = np.reshape(ey_magnitude_train, (ey_magnitude_train.shape[0], ey_magnitude_train.shape[1]*ey_magnitude_train.shape[2]))
    ey_magnitude_train_normalized = preprocessing.normalize(ey_magnitude_train, norm='l2')
    ez_magnitude_train = np.copy(train_set_X[:, 4, :, :])
    ez_magnitude_train = np.reshape(ez_magnitude_train, (ez_magnitude_train.shape[0], ez_magnitude_train.shape[1]*ez_magnitude_train.shape[2]))
    ez_magnitude_train_normalized = preprocessing.normalize(ez_magnitude_train, norm='l2')


    model_cross_val_scores = np.zeros((6, 2))
    m_train = Y_train_class[:, 0]
    n_train = Y_train_class[:, 1]
    modes_train = Y_train_class[:, 2]

    print("Train TE/TM separately from m and n")
    sgd_clf = SGDClassifier()
    sgd_ave = np.average(cross_val_score(sgd_clf, ez_magnitude_train_normalized, modes_train, cv=3, scoring="accuracy"))
    print(f"TE/TM SGD Average Score on 3 Cross-Evaluaitons: {sgd_ave}")

    ### ML model trains on the Ex
    y_multilabel = np.c_[m_train, n_train]

    print("neighbors.KNeighborsClassifier:")
    knn_clf = KNeighborsClassifier()
    y_train_knn_pred = cross_val_predict(knn_clf, ex_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_knn_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_knn_pred[:, 1], average="macro"))
    model_cross_val_scores[0, 0] = f1_score(y_multilabel[:, 0], y_train_knn_pred[:, 0], average="macro")
    model_cross_val_scores[0, 1] = f1_score(y_multilabel[:, 1], y_train_knn_pred[:, 1], average="macro")

    print("tree.DecisionTreeClassifier:")
    y_train_tree_pred = cross_val_predict(tree_class, ex_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    model_cross_val_scores[1, 0] = f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro")
    model_cross_val_scores[1, 1] = f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro")

    print("tree.ExtraTreeClassifier")
    from sklearn.tree import ExtraTreeClassifier
    tree_class = ExtraTreeClassifier()
    y_train_tree_pred = cross_val_predict(tree_class, ex_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    model_cross_val_scores[2, 0] = f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro")
    model_cross_val_scores[2, 1] = f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro")

    print("ensemble.ExtraTreeClassifier")
    from sklearn.ensemble import ExtraTreesClassifier
    tree_class = ExtraTreesClassifier()
    y_train_tree_pred = cross_val_predict(tree_class, ex_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro"))
    model_cross_val_scores[3, 0] = f1_score(y_multilabel[:, 0], y_train_tree_pred[:, 0], average="macro")
    model_cross_val_scores[3, 1] = f1_score(y_multilabel[:, 1], y_train_tree_pred[:, 1], average="macro")

    print("ensemble.RandomForestClassifier")
    from sklearn.ensemble import RandomForestClassifier
    forest_class = RandomForestClassifier()
    y_train_forest_pred = cross_val_predict(forest_class, ex_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_forest_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_forest_pred[:, 1], average="macro"))
    model_cross_val_scores[4, 0] = f1_score(y_multilabel[:, 0], y_train_forest_pred[:, 0], average="macro")
    model_cross_val_scores[4, 1] = f1_score(y_multilabel[:, 1], y_train_forest_pred[:, 1], average="macro")

    print("neighbors.RadiusNeighborsClassifier")
    from sklearn.neighbors import RadiusNeighborsClassifier
    radius_class = RadiusNeighborsClassifier(radius=5.0)
    y_train_radius_pred = cross_val_predict(radius_class, ex_magnitude_train_normalized, y_multilabel, cv=3)
    print("\tm f1 score: ", f1_score(y_multilabel[:, 0], y_train_radius_pred[:, 0], average="macro"))
    print("\tn f1 score: ", f1_score(y_multilabel[:, 1], y_train_radius_pred[:, 1], average="macro"))
    model_cross_val_scores[5, 0] = f1_score(y_multilabel[:, 0], y_train_radius_pred[:, 0], average="macro")
    model_cross_val_scores[5, 1] = f1_score(y_multilabel[:, 1], y_train_radius_pred[:, 1], average="macro")
    print("\n")

    return sgd_ave, model_cross_val_scores