# Machine Learning for Detection of Modal Configurations in Rectangular Waveguides
### Author: Jennifer E. Houle
#### Date: 5/4/2022

This code was written for a class project in the course entitled ECE 504: 
"Machine Learning for Electromagnetics" that was taught in Spring 2022 by 
Prof. Zadehgol at the University of Idaho in Moscow, Idaho.

## Overview
This project generates the x-y cross-section of a rectangular waveguide at z = 0 using [1] for the E and H
fields in the x, y, and z directions. This is done for a specified selection of modes with a designated 
number of samples generated per mode. The same amount of data is generated for both TE and TM. 
Noise (exponential or Gaussian) can then be added.

A sweep can be done of either the number of samples per mode or of the amount of noise added.

A Machine Learning algorithm then identifies the mode (m, n and whether it is TE or TM).

## Licensing
License file included. The Machine Learning code was based on code learned from [2]. 
License for the Machine Learning code implementation is subject to [[3](https://github.com/ageron/handson-ml2/blob/master/LICENSE)].

## Files

- `final_project_main.py`: run the program to generate waveguide cross-sectional data, add noise, and run a ML algorithm
to find the mode information. This has the option to generate the desired data across different sample sizes.
- `compare_noise_main.py`: run the program to generate waveguide cross-sectional data, add noise, and run a ML algorithm
to find the mode information. This has the option to generate the desired data across different amounts to noise.
- `generate_waveguide_data.py`: generate the waveguide data based on the equations [1] and add noise if desired.
- `plot_fields.py`: plot the waveguide cross-sections
- `waveguide_decision_tree_algorithm.py`: use the Decision Tree Machine Learning algorithm to find the mode information 
from an x-y cross-section. Some code based on [2].
- `waveguide_random_forest_algorithm.py`: use the Random Forest Machine Learning algorithm to find the mode information 
from an x-y cross-section. Some code based on [2].
- `compare_machine_learning_defaults.py`: use the 6 multi-class, multi-label classifiers to find the mode information 
from an x-y cross-section. Some code based on [2].
  - There are several variations of this available in the file to run the classifer based on:
    - `compare_machine_learning_2_algorithms`: Ex only for m, n classification; an SGD classifier is used for TE/TM classification
    - `compare_machine_learning_defaults`: Ex, Ez concatenated to classify m, n, and TE/TM
    - `compare_machine_learning_all_e_defaults`: Ex, Ey, Ez concatenated to classify m, n, and TE/TM
- `classifier_compare_graphs.py`: generate plots to compare the various classifiers across numbers of samples per mode.
- `noise_compare_graphs.py`: generate plots to compare the various classifiers across the amount of noise added.

## Code
See Files above.

## Run instructions
1. Run `final_project_main.py` to vary the number of samples per mode
2. Run `classifier_compare_graphs.py` to generate plots comparing the results form `final_project_main.py`

Alternatively,
1. Run `compare_noise_main.py` to compare the amount of noise added
2. Run `noise_compare_graphs.py` to generate plots comparing the results form `compare_noise_main.py`


## Input parameters
The project is currently set up with the following parameters easy to adjust:
- number_of_points_per_direction: the number of cells in which the x and y 
- direction are each divided
- number_of_m: the number of m values. Modes will be generated for 
  - m = 1
  - m = 2
  - ... 
  - m = number_of_m
- number_of_n: the number of n values. Modes will be generated for 
  - n = 1
  - n = 2
  - ... 
  - n = number_of_n
- samples: the number of samples at each mode. It can be an array, in which 
case results are generated for each number of samples within the array.
- exponential_noise: If true, exponential noise; if false, Gaussian noise
- scale: [Beta](https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html) 
for exponential noise, 
[variance](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html) for Gaussian noise
- mean: this is only used in generating Gaussian noise

## Output parameters
1. Visual examples cross-sections of the waveguide are generated and saved as `waveguide_data.npz`. The number generated can be adjusted by adjusting the range 
in `for config in range(2):` within `final_project_main.py`.
2. `ml_dectree_scores.npy`: Decision Tree Classifier's results (with one algorithm trained on Ex, Ey, and Ez magnitudes) 
stored in a row per number of samples per mode. Order stored is: 
   1. f1 score for m
   2. f1 score for n
   3. f1 score for whether it is TE/TM
   4. test data's error rate for m
   5. test data's error rate for n
   6. test data's error rate for TE/TM selection
   7. test data's error rate for any error.
3. `ml_dectree_separate_scores.npy`: Decision Tree Classifier's results (with 3 algorithms trained on Ex, Ey, and Ez magnitudes separately for each mode component)
stored in a row per number of samples per mode. Order stored is: 
   1. f1 score for m
   2. f1 score for n
   3. f1 score for whether it is TE/TM
   4. test data's error rate for m
   5. test data's error rate for n
   6. test data's error rate for TE/TM selection
   7. test data's error rate for any error.
5. `ml_randomforest_scores.npy`: Same as `ml_dectree_scores.npy` but using the Random Forest Classifier.
6. `ml_ex_ez_cross_val_scores.npy`: various classifiers' f1 results stored per number of samples per mode.
   ((number_of_samples_per_mode, classifiers' results, results))
   - Order of classifiers is:
     - neighbors.KNeighborsClassifier
     - tree.DecisionTreeClassifier
     - tree.ExtraTreeClassifier
     - ensemble.ExtraTreesClassifier
     - ensemble.RandomForestClassifier
     - neighbors.RadiusNeighborsClassifier
   - Order of data (f1 scores):
     - m
     - n
     - whether it is TE/TM
7. `ml_ex_ey_ez_cross_val_scores.npy`: same data as `ml_ex_ez_cross_val_scores.npy`, but trained on Ex, Ey, and Ez
8. `ml_ex_cross_val_scores.npy`: same data as `ml_ex_ez_cross_val_scores.npy`, but trained on Ex only and no TE/TM data included.
9. `ml_sgd_scores.npy` contains the corresponding to `ml_ex_cross_val_scores.npy` for the 
TE/TM from the SGD classifier
10. `samples.npy`: the array of the number of samples per mode

Note a suffix may be added to the `.npy` files above to differentiate different results (such as making one run `_exp` for exponential noise).
It is called `file_suffix` in `final_project_main.py`.

## Usage
The basic program is straight forward, with only a few different inputs available.

The number of samples per mode can be an array to see the effect of more samples on the scores/error rates.

Within `generate_waveguide_data.py` the waveguide parameters can be easily adjusted from
the current configuration (a = 1.07 cm, b = 0.43 cm, A = B = 1, using Teflon).

The classifiers are used for creating the cross-validation scores.
A customized error rate is output for each of the mode components individually 
and in aggregate for the Decision Tree and Random Forest.

The DecisionTreeClassifier may be adjusted using hyperparameters 
or a different classifier used within
`waveguide_decision_tree_algorithm.py`

The RandomForestClassifier may be adjusted using hyperparameters 
or a different classifier used within
`waveguide_random_forest_algorithm.py`

The various classifiers use defaults when multiple are tested at the same time, 
with the exception of `radius_class = RadiusNeighborsClassifier(radius=5.0)` 
since the default did not find a solution. The hyperparameters of these can be changed within
`compare_machine_learning_defaults.py` by finding the various classifiers.

The classifiers can be visually compared by running `classifier_compare_graphs.py`.

## Python version info
Python 3.9.7
See environment.yml for complete information.

## References
[1]	D. Pozar. Microwave Engineering, 4th ed. New York, USE: Wiley 2011, ch 3.

[2] A. Géron, Hands-on Machine Learning with Scikit-Learn, Keras & TensforFlow, 2nd ed. Sebastopol, 
CA, USA:O’Reilly 2019, ch. 1-3, pp. 1-108.

[3] https://github.com/ageron/handson-ml2/blob/master/LICENSE

