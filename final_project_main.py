""" final_project_main.py

Generate cross-sectional waveguide data (generate samples for various modes including TE/TM, m, and n)
Add noise to the data (if desired)
Run for the selection of sample sizes
Run Machine Learning algorithm to identify the mode (results given per selection of sample sizes for various
classifiers)

Author: Jennifer Houle
Date: 4/22/2022

"""

import numpy as np

from generate_waveguide_data import generate_data_configurations, generate_output_data
from plot_fields import plot_e_fields, plot_h_fields, plot_ex_field
from waveguide_decision_tree_algorithm import decision_tree_machine_learning, decision_tree_machine_learning_separate
from waveguide_random_forest_algorithm import random_forest_machine_learning
from compare_machine_learning_defaults import compare_machine_learning_defaults, \
    compare_machine_learning_all_e_defaults, compare_machine_learning_2_algorithms

# Data that is adjusted
number_of_points_per_direction = 25
number_of_m = 3
number_of_n = 3
samples = np.array([5, 10, 25]) # This is the number of samples at each TM/TE and m/n combination
# samples = np.array([5, 10, 25, 50, 100, 200, 400, 800, 1_000]) # This is the number of samples at each TM/TE and m/n combination

# Arrays in which to save the ML results
ml_dectree_scores = np.zeros((samples.shape[0], 7))
ml_dectree_separate_scores = np.zeros((samples.shape[0], 7))
ml_randomforest_scores = np.zeros((samples.shape[0], 7))
ml_ex_ey_ez_cross_val_scores = np.zeros((samples.shape[0], 6, 3))
ml_ex_ez_cross_val_scores = np.zeros((samples.shape[0], 6, 3))
ml_ex_cross_val_scores = np.zeros((samples.shape[0], 6, 2))
ml_sgd_scores = np.zeros((samples.shape[0]))

# Noise parameters
exponential_noise = True    # If true, exponential noise
                            # If false, Gaussian noise
scale = np.sqrt(5.0)        # Beta for exponential noise, variance for Gaussian
mean = 0.0                  # This only has an effect on Gaussian noise

file_suffix = '_exp'  # This can be used to specify the data file names saved at the end of the program

for data in range(samples.shape[0]):
    number_of_samples=samples[data]
    print(f"Run for data samples per mode = {number_of_samples}")

    # Generate the configurations
    input_data = generate_data_configurations(number_of_m, number_of_n, number_of_samples)

    # The input data is shuffled for future Machine Learning
    np.random.shuffle(input_data)

    # Generate the E, H cross-sectional field data in each direction
    output_data, waveguide = generate_output_data(input_data, exponential_noise, scale, mean, number_of_points_per_direction)

    # Plot - only do this for a few configurations because it gets slow!
    for config in range(2):
        plot_e_fields(output_data[config, :, :], input_data[config, 2], int(input_data[config, 0]), int(input_data[config, 1]), waveguide)
        plot_h_fields(output_data[config, :, :], input_data[config, 2], int(input_data[config, 0]), int(input_data[config, 1]), waveguide)
        plot_ex_field(output_data[config, :, :], input_data[config, 2], int(input_data[config, 0]), int(input_data[config, 1]), waveguide)

    np.savez("waveguide_data", input_data, output_data)

    # Evaluation of Decision Tree ML Algorithm:
    ml_dectree_scores[data, :] = decision_tree_machine_learning()

    # Evaluation of Decision Tree ML Algorithm:
    ml_dectree_separate_scores[data, :] = decision_tree_machine_learning_separate()

    # Evaluation of Random Forest ML Algorithm:
    ml_randomforest_scores[data, :] = random_forest_machine_learning()

    # Evaluation of Various ML Algorithms on Ex and Ey only:
    ml_ex_ez_cross_val_scores[data, :] = compare_machine_learning_defaults()

    # Evaluation of Various ML Algorithms on Ex, Ey and Ez:
    ml_ex_ey_ez_cross_val_scores[data, :] = compare_machine_learning_all_e_defaults()

    # Evaluation of Various ML Algorithms on Ex for m and n; SGD on Ex for TE/TM:
    ml_sgd_scores[data], ml_ex_cross_val_scores[data, :] = compare_machine_learning_2_algorithms()

np.save(f"samples{file_suffix}", samples)
np.save(f"ml_dectree_scores{file_suffix}", ml_dectree_scores)
np.save(f"ml_dectree_separate_scores{file_suffix}", ml_dectree_separate_scores)
np.save(f"ml_randomforest_scores{file_suffix}", ml_randomforest_scores)
np.save(f"ml_ex_ez_cross_val_scores{file_suffix}", ml_ex_ez_cross_val_scores)
np.save(f"ml_ex_ey_ez_cross_val_scores{file_suffix}", ml_ex_ey_ez_cross_val_scores)
np.save(f"ml_ex_cross_val_scores{file_suffix}", ml_ex_cross_val_scores)
np.save(f"ml_sgd_scores{file_suffix}", ml_sgd_scores)

