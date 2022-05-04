""" noise_compare_graphs.py

Generate graphs to visualize the performance of various classifiers when varying the amount of noise.

Data was generated with compare_noise_main.py

4/25/2022
"""

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'legend.fontsize': 14})

scale_array = np.load("scale_array_100.npy")
samples = 100

ml_dectree_scores = np.load(f"ml_dectree_scores_vary_scale_{samples}.npy")
plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_dectree_scores[:, 0], label='m')
plt.plot(scale_array, ml_dectree_scores[:, 1], label='n')
plt.plot(scale_array, ml_dectree_scores[:, 2], label='TE/TM')
plt.title('Decision Tree $f_1$ Scores (Training on $E_x$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_dectree_scores[:, 3], label='m')
plt.plot(scale_array, ml_dectree_scores[:, 4], label='n')
plt.plot(scale_array, ml_dectree_scores[:, 5], label='TE/TM')
plt.plot(scale_array, ml_dectree_scores[:, 6], label='m, n, or TE/TM')
plt.title('Decision Tree Error Rate (Training on $E_x$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("Error Rate")
plt.legend()
plt.tight_layout()
plt.show()


ml_randomforest_scores = np.load(f"ml_randomforest_scores_vary_scale_{samples}.npy")

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_randomforest_scores[:, 0], label='m')
plt.plot(scale_array, ml_randomforest_scores[:, 1], label='n')
plt.plot(scale_array, ml_randomforest_scores[:, 2], label='TE/TM')
plt.title('Random Forest $f_1$ Scores (Training on $E_x$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_randomforest_scores[:, 3], label='m')
plt.plot(scale_array, ml_randomforest_scores[:, 4], label='n')
plt.plot(scale_array, ml_randomforest_scores[:, 5], label='TE/TM')
plt.plot(scale_array, ml_randomforest_scores[:, 6], label='m, n, or TE/TM')
plt.title('Random Forest Error Rate (Training on $E_x$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("Error Rate")
plt.legend()
plt.tight_layout()
plt.show()

ml_ex_ez_cross_val_scores = np.load(f"ml_ex_ez_cross_val_scores_vary_scale_{samples}.npy")

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 0, 0], label='KNeighbors')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 1, 0], label='DecisionTree')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 2, 0], label='tree.ExtraTree')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 3, 0], label='ensemble.ExtraTrees')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 4, 0], label='RandomForest')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 5, 0], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $m$ Scores (Training on $E_x$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 0, 1], label='KNeighbors')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 1, 1], label='DecisionTree')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 2, 1], label='tree.ExtraTree')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 3, 1], label='ensemble.ExtraTrees')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 4, 1], label='RandomForest')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 5, 1], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $n$ Scores (Training on $E_x$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$n$ $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 0, 2], label='KNeighbors')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 1, 2], label='DecisionTree')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 2, 2], label='tree.ExtraTree')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 3, 2], label='ensemble.ExtraTrees')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 4, 2], label='RandomForest')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 5, 2], label='RadiusNeighbors')
plt.title('Classifier $f_1$ TE/TM Scores (Training on $E_x$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("TE/TM $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

ml_ex_ey_ez_cross_val_scores = np.load(f"ml_ex_ey_ez_cross_val_scores_vary_scale_{samples}.npy")

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 0, 0], label='KNeighbors')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 1, 0], label='DecisionTree')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 2, 0], label='tree.ExtraTree')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 3, 0], label='ensemble.ExtraTrees')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 4, 0], label='RandomForest')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 5, 0], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $m$ Scores (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 0, 1], label='KNeighbors')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 1, 1], label='DecisionTree')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 2, 1], label='tree.ExtraTree')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 3, 1], label='ensemble.ExtraTrees')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 4, 1], label='RandomForest')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 5, 1], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $n$ Scores (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$n$ $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 0, 2], label='KNeighbors')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 1, 2], label='DecisionTree')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 2, 2], label='tree.ExtraTree')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 3, 2], label='ensemble.ExtraTrees')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 4, 2], label='RandomForest')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 5, 2], label='RadiusNeighbors')
plt.title('Classifier $f_1$ TE/TM Scores (Training on $E_x$, $E_y$, $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("TE/TM $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

ml_ex_cross_val_scores = np.load(f"ml_ex_cross_val_scores_vary_scale_{samples}.npy")
plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_cross_val_scores[:, 0, 0], label='KNeighbors')
plt.plot(scale_array, ml_ex_cross_val_scores[:, 1, 0], label='DecisionTree')
plt.plot(scale_array, ml_ex_cross_val_scores[:, 2, 0], label='tree.ExtraTree')
plt.plot(scale_array, ml_ex_cross_val_scores[:, 3, 0], label='ensemble.ExtraTrees')
plt.plot(scale_array, ml_ex_cross_val_scores[:, 4, 0], label='RandomForest')
plt.plot(scale_array, ml_ex_cross_val_scores[:, 5, 0], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $m$ Scores (Training on $E_x$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 0, 1], label='KNeighbors')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 1, 1], label='DecisionTree')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 2, 1], label='tree.ExtraTree')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 3, 1], label='ensemble.ExtraTrees')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 4, 1], label='RandomForest')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 5, 1], label='RadiusNeighbors')
plt.title('Classifier $f_1$ $n$ Scores (Training on $E_x$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$n$ $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()


ml_sgd_scores = np.load(f"ml_sgd_scores_vary_scale_{samples}.npy")
plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_sgd_scores)
plt.title('SGD Ave. TE/TM Scores (Training on $E_z$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("TE/TM Ave. Cross. Val. Score")
plt.tight_layout()
plt.show()

# Compare Random Forest performance based on inputs used.
plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_cross_val_scores[:, 4, 0], label='Trained on $E_x$')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 4, 0], label='Trained on $E_x$, $E_z$')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 4, 0], label='Trained on $E_x$, $E_y$, $E_z$')
plt.title('Random Forest Classifier $f_1$ $m$ Scores (Training on $E_x$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(scale_array, ml_ex_cross_val_scores[:, 4, 1], label='Trained on $E_x$')
plt.plot(scale_array, ml_ex_ez_cross_val_scores[:, 4, 1], label='Trained on $E_x$, $E_z$')
plt.plot(scale_array, ml_ex_ey_ez_cross_val_scores[:, 4, 1], label='Trained on $E_x$, $E_y$, $E_z$')
plt.title('Random Forest Classifier $f_1$ $n$ Scores (Training on $E_x$)')
plt.xlabel("Scale of Noise Added")
plt.ylabel("$n$ $f_1$ Score")
plt.legend()
plt.tight_layout()
plt.show()