""" classifier_compare_graphs.py

Generate graphs to visualize the performance of various classifiers when varying the number of samples per mode.

Data was generated with final_project_main.py

4/21/2022
"""

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'legend.fontsize': 14})

file_suffix = '_exp'

samples = np.load(f"samples{file_suffix}.npy")

ml_dectree_scores = np.load(f"ml_dectree_scores{file_suffix}.npy")

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_dectree_scores[:, 0], label='m')
plt.plot(samples, ml_dectree_scores[:, 1], label='n')
plt.plot(samples, ml_dectree_scores[:, 2], label='TE/TM')
plt.title('Decision Tree $f_1$ Scores (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_dectree_scores[:, 0], 'g', label='m (Training)')
plt.plot(samples, 1 - ml_dectree_scores[:, 3], 'g--', label='m (Test)')
plt.plot(samples, ml_dectree_scores[:, 1], 'b', label='n (Training)')
plt.plot(samples, 1 - ml_dectree_scores[:, 4], 'b--', label='n (Test)')
plt.plot(samples, ml_dectree_scores[:, 2], 'm', label='TE/TM (Training)')
plt.plot(samples, 1 - ml_dectree_scores[:, 5], 'm--', label='TE/TM (Test)')
plt.title('Decision Tree Accuracy (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$/Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_dectree_scores[:, 3], label='m')
plt.plot(samples, ml_dectree_scores[:, 4], label='n')
plt.plot(samples, ml_dectree_scores[:, 5], label='TE/TM')
plt.plot(samples, ml_dectree_scores[:, 6], label='m, n, or TE/TM')
plt.title('Decision Tree Error Rate (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("Error Rate")
plt.legend()
plt.tight_layout()
plt.show()

ml_dectree_separate_scores = np.load(f"ml_dectree_separate_scores{file_suffix}.npy")

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_dectree_separate_scores[:, 0], label='m')
plt.plot(samples, ml_dectree_separate_scores[:, 1], label='n')
plt.plot(samples, ml_dectree_separate_scores[:, 2], label='TE/TM')
plt.title('Decision Tree $f_1$ Scores (Separate $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_dectree_separate_scores[:, 0], 'g', label='m (Training on |$E_y$|)')
plt.plot(samples, 1 - ml_dectree_separate_scores[:, 3], 'g--', label='m (Test)')
plt.plot(samples, ml_dectree_separate_scores[:, 1], 'b', label='n (Training on |$E_x$|)')
plt.plot(samples, 1 - ml_dectree_separate_scores[:, 4], 'b--', label='n (Test)')
plt.plot(samples, ml_dectree_separate_scores[:, 2], 'm', label='TE/TM (Training on |$E_z$|)')
plt.plot(samples, 1 - ml_dectree_separate_scores[:, 5], 'm--', label='TE/TM (Test)')
plt.title('Decision Tree Accuracy')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$/Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


ml_randomforest_scores = np.load(f"ml_randomforest_scores{file_suffix}.npy")

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_randomforest_scores[:, 0], label='m')
plt.plot(samples, ml_randomforest_scores[:, 1], label='n')
plt.plot(samples, ml_randomforest_scores[:, 2], label='TE/TM')
plt.title('Random Forest $f_1$ Scores (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_randomforest_scores[:, 0], 'g', label='m (Training)')
plt.plot(samples, 1 - ml_randomforest_scores[:, 3], 'g--', label='m (Test)')
plt.plot(samples, ml_randomforest_scores[:, 1], 'b', label='n (Training)')
plt.plot(samples, 1 - ml_randomforest_scores[:, 4], 'b--', label='n (Test)')
plt.plot(samples, ml_randomforest_scores[:, 2], 'm', label='TE/TM (Training)')
plt.plot(samples, 1 - ml_randomforest_scores[:, 5], 'm--', label='TE/TM (Test)')
plt.title('Random Forest Accuracy (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$/Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_randomforest_scores[:, 3], label='m')
plt.plot(samples, ml_randomforest_scores[:, 4], label='n')
plt.plot(samples, ml_randomforest_scores[:, 5], label='TE/TM')
plt.plot(samples, ml_randomforest_scores[:, 6], label='m, n, or TE/TM')
plt.title('Random Forest Error Rate (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("Error Rate")
plt.legend()
plt.tight_layout()
plt.show()

ml_ex_ez_cross_val_scores = np.load(f"ml_ex_ez_cross_val_scores{file_suffix}.npy")

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 0, 0], label='KNeighbors')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 1, 0], label='DecisionTree')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 2, 0], label='tree.ExtraTree')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 3, 0], label='ensemble.ExtraTrees')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 4, 0], label='RandomForest')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 5, 0], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $m$ Scores (Training on $E_x$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 0, 1], label='KNeighbors')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 1, 1], label='DecisionTree')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 2, 1], label='tree.ExtraTree')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 3, 1], label='ensemble.ExtraTrees')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 4, 1], label='RandomForest')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 5, 1], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $n$ Scores (Training on $E_x$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$n$ $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 0, 2], label='KNeighbors')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 1, 2], label='DecisionTree')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 2, 2], label='tree.ExtraTree')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 3, 2], label='ensemble.ExtraTrees')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 4, 2], label='RandomForest')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 5, 2], label='RadiusNeighbors')
plt.title('Classifier $f_1$ TE/TM Scores (Training on $E_x$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("TE/TM $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

ml_ex_ey_ez_cross_val_scores = np.load(f"ml_ex_ey_ez_cross_val_scores{file_suffix}.npy")

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 0, 0], label='KNeighbors')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 1, 0], label='DecisionTree')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 2, 0], label='tree.ExtraTree')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 3, 0], label='ensemble.ExtraTrees')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 4, 0], label='RandomForest')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 5, 0], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $m$ Scores (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 0, 1], label='KNeighbors')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 1, 1], label='DecisionTree')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 2, 1], label='tree.ExtraTree')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 3, 1], label='ensemble.ExtraTrees')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 4, 1], label='RandomForest')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 5, 1], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $n$ Scores (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$n$ $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 0, 2], label='KNeighbors')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 1, 2], label='DecisionTree')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 2, 2], label='tree.ExtraTree')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 3, 2], label='ensemble.ExtraTrees')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 4, 2], label='RandomForest')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 5, 2], label='RadiusNeighbors')
plt.title('Classifier $f_1$ TE/TM Scores (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("TE/TM $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

ml_ex_cross_val_scores = np.load(f"ml_ex_cross_val_scores{file_suffix}.npy")
plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_cross_val_scores[:, 0, 0], label='KNeighbors')
plt.plot(samples, ml_ex_cross_val_scores[:, 1, 0], label='DecisionTree')
plt.plot(samples, ml_ex_cross_val_scores[:, 2, 0], label='tree.ExtraTree')
plt.plot(samples, ml_ex_cross_val_scores[:, 3, 0], label='ensemble.ExtraTrees')
plt.plot(samples, ml_ex_cross_val_scores[:, 4, 0], label='RandomForest')
plt.plot(samples, ml_ex_cross_val_scores[:, 5, 0], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $m$ Scores (Training on $E_x$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 0, 1], label='KNeighbors')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 1, 1], label='DecisionTree')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 2, 1], label='tree.ExtraTree')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 3, 1], label='ensemble.ExtraTrees')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 4, 1], label='RandomForest')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 5, 1], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $n$ Scores (Training on $E_x$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$n$ $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()


ml_sgd_scores = np.load(f"ml_sgd_scores{file_suffix}.npy")
plt.figure(figsize=(8, 4))
plt.plot(samples, ml_sgd_scores)
plt.title('SGD Ave. TE/TM Scores (Training on $E_z$)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("TE/TM Ave. Cross. Val. Score")
plt.tight_layout()
plt.show()

# Compare Random Forest performance based on inputs used.
plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_cross_val_scores[:, 4, 0], label='Trained on $E_x$')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 4, 0], label='Trained on $E_x$, $E_z$')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 4, 0], label='Trained on $E_x$, $E_y$, $E_z$')
plt.title('Random Forest Classifier $f_1$ $m$ Scores')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_ex_cross_val_scores[:, 4, 1], label='Trained on $E_x$')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 4, 1], label='Trained on $E_x$, $E_z$')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 4, 1], label='Trained on $E_x$, $E_y$, $E_z$')
plt.title('Random Forest Classifier $f_1$ $n$ Scores')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("$n$ $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(samples, ml_sgd_scores, label='SGD trained on $E_z$')
plt.plot(samples, ml_ex_ez_cross_val_scores[:, 4, 1], label='Random Forest trained on $E_x$, $E_z$')
plt.plot(samples, ml_ex_ey_ez_cross_val_scores[:, 4, 1], label='Random Forest trained on $E_x$, $E_y$, $E_z$')
plt.title('Classifier Scores (TE/TM)')
plt.xlabel("Number of Samples per Mode")
plt.ylabel("TE/TM Score")
plt.legend()
plt.tight_layout()
plt.show()