""" generate_waveguide_data.py

Generate the waveguide data based on the equations in Pozar (see readme for reference).
Add noise

Author: Jennifer Houle
Date: 4/22/2022

"""

import numpy as np
import cmath

class WaveguideParameters:
    def __init__(self, number_of_points_per_direction):
        """
        Save the waveguide parameters into one class
        :param number_of_points_per_direction: this is the number of cells in the x and y direction of the cross-section
        of the waveguide
        """
        mu_0 = 4 * np.pi * 1e-7
        self.mu = 1.0 * mu_0
        epsilon_0 = 8.85418782e-12
        self.epsilon = 2.08 * epsilon_0
        self.a = 1.07e-2
        self.b = 0.43e-2
        self.A_amp = 1
        self.B_amp = 1
        self.x = np.linspace(self.a / number_of_points_per_direction, self.a, number_of_points_per_direction)
        self.y = np.linspace(self.b / number_of_points_per_direction, self.b, number_of_points_per_direction)
        self.z = 0
        self.number_of_points_per_direction = number_of_points_per_direction

def generate_data_configurations(number_of_m, number_of_n, number_of_samples):
    """
    Generate an array containing all the combinations of m, n, and TE/TM
    :param number_of_m: number of m values, beginning with 1. (So 3 results in m=1, m=2, m=3)
    :param number_of_n: number of n values, beginning with 1. (So 3 results in n=1, n=2, n=3)
    :param number_of_samples: number of samples for each mode (TE/TM, m, and n combination)
    :return:
        input_data: array of inputs
    """
    number_of_configurations = number_of_m * number_of_n * number_of_samples * 2  # The 2 accounts for TE/TM
    input_data = np.zeros((number_of_configurations, 3))

    # This generates the configurations used to calculate E/H fields (which will be labels in the ML data set)
    index = 0
    for m in range(1, number_of_m + 1):
        for n in range(1, number_of_n + 1):
            for f in range(number_of_samples):
                input_data[index, 0] = m
                input_data[index, 1] = n
                input_data[index, 2] = 0  # 'TM'
                input_data[index + number_of_m * number_of_n * number_of_samples, 0] = m
                input_data[index + number_of_m * number_of_n * number_of_samples, 1] = n
                input_data[index + number_of_m * number_of_n * number_of_samples, 2] = 1  # 'TE'
                index += 1

    return input_data

def generate_e_and_h_data(input_data, freq, waveguide):
    """
    Generate the complete E and H field data (magnitude and phase) for the waveguide in the x, y, and z directions
    :param input_data: array of input mode configurations on which to generate waveguide data
    :param freq: frequency at which the mode is being generated
    :param waveguide: class containing the waveguide parameters
    :return: output_data_per_input: the magnitude and phase for Ex, Ey, Ez, Hx, Hy, Hz
    """
    m = input_data[0]
    n = input_data[1]
    omega = 2 * np.pi * freq
    mode_TE = input_data[2]
    k = omega * np.sqrt(waveguide.mu * waveguide.epsilon)
    k_c = np.sqrt(np.power(m * np.pi / waveguide.a, 2) + np.power(n * np.pi / waveguide.b, 2))
    if k_c == 0:
        k_c = 1e-15
    beta = cmath.sqrt(np.power(k, 2) - np.power(k_c, 2))
    output_data_per_input = np.zeros((12, waveguide.number_of_points_per_direction, waveguide.number_of_points_per_direction))

    if mode_TE == True:
        for y_value in range(waveguide.number_of_points_per_direction):
            e_x = 1j * omega * waveguide.mu * n * np.pi / (np.power(k_c, 2) * waveguide.b) * waveguide.A_amp * np.cos(m * np.pi * waveguide.x / waveguide.a) * np.sin(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[0, y_value, :] = np.absolute(e_x)
            output_data_per_input[1, y_value, :] = np.angle(e_x)

            e_y = -1j * omega * waveguide.mu * m * np.pi / (np.power(k_c, 2) * waveguide.a) * waveguide.A_amp * np.sin(m * np.pi * waveguide.x / waveguide.a) * np.cos(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[2, y_value, :] = np.absolute(e_y)
            output_data_per_input[3, y_value, :] = np.angle(e_y)

            e_z = 0 * waveguide.x
            output_data_per_input[4, y_value, :] = np.absolute(e_z)
            output_data_per_input[5, y_value, :] = np.angle(e_z)

            h_x = -1j * beta * m * np.pi / (np.power(k_c, 2) * waveguide.a) * waveguide.A_amp * np.sin(m * np.pi * waveguide.x / waveguide.a) * np.cos(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[6, y_value, :] = np.absolute(h_x)
            output_data_per_input[7, y_value, :] = np.angle(h_x)

            h_y = 1j * beta * n * np.pi / (np.power(k_c, 2) * waveguide.b) * waveguide.A_amp * np.cos(m * np.pi * waveguide.x / waveguide.a) * np.sin(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[8, y_value, :] = np.absolute(h_y)
            output_data_per_input[9, y_value, :] = np.angle(h_y)

            h_z = waveguide.A_amp * np.cos(m * np.pi * waveguide.x / waveguide.a) * np.cos(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[10, y_value, :] = np.absolute(h_z)
            output_data_per_input[11, y_value, :] = np.angle(h_z)
    else:
        for y_value in range(waveguide.number_of_points_per_direction):
            e_x = -1j * beta * m * np.pi / (np.power(k_c, 2) * waveguide.a) * waveguide.B_amp * np.cos(m * np.pi * waveguide.x / waveguide.a) * np.sin(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[0, y_value, :] = np.absolute(e_x)
            output_data_per_input[1, y_value, :] = np.angle(e_x)

            e_y = -1j * beta * n * np.pi / (np.power(k_c, 2) * waveguide.b) * waveguide.B_amp * np.sin(m * np.pi * waveguide.x / waveguide.a) * np.cos(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[2, y_value, :] = np.absolute(e_y)
            output_data_per_input[3, y_value, :] = np.angle(e_y)

            e_z = waveguide.B_amp * np.sin(m * np.pi * waveguide.x / waveguide.a) * np.sin(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[4, y_value, :] = np.absolute(e_z)
            output_data_per_input[5, y_value, :] = np.angle(e_z)

            h_x = 1j * omega * waveguide.epsilon * n * np.pi / (np.power(k_c, 2) * waveguide.b) * waveguide.B_amp * np.sin(m * np.pi * waveguide.x / waveguide.a) * np.cos(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[6, y_value, :] = np.absolute(h_x)
            output_data_per_input[7, y_value, :] = np.angle(h_x)

            h_y = -1j * omega * waveguide.epsilon * m * np.pi / (np.power(k_c, 2) * waveguide.a) * waveguide.B_amp * np.cos(m * np.pi * waveguide.x / waveguide.a) * np.sin(n * np.pi * waveguide.y[y_value] / waveguide.b) * np.exp(-1j * beta * waveguide.z)
            output_data_per_input[8, y_value, :] = np.absolute(h_y)
            output_data_per_input[9, y_value, :] = np.angle(h_y)

            h_z = 0 * waveguide.x
            output_data_per_input[10, y_value, :] = np.absolute(h_z)
            output_data_per_input[11, y_value, :] = np.angle(h_z)

    return output_data_per_input

def add_exponential_noise(data, scale, percentage_of_max):
    """
    Add exponential noise to the data
    :param data: np array of the calculated value at each cross-sectional location
    :param scale: scale parameter Beta = 1/lambda in the probability density function (https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html)
    :param percentage_of_max: if true, scale is scaled by the maximum data value. Otherwise, it is an absolute value
    :return: data + exponential noise
    """
    row,col = data.shape
    if percentage_of_max:
        maximum = np.max(data)
        if maximum != 0:
            scale = scale * maximum
    exp_noise = np.random.exponential(scale, (row, col))
    exp_noise = exp_noise.reshape(row, col)
    noisy_image = data + exp_noise
    return noisy_image

def add_gaussian_noise(data, variance, mean, percentage_of_max):
    """
    Add Gaussian noise to the data
    :param data: np array of the calculated value at each cross-sectional location
    :param variance: variance for the noise to be added
    :param mean: mean of the noise to be added
    :param percentage_of_max: if true, variance is scaled by the maximum data value. Otherwise, it is an absolute value
    :return: data + gaussian noise
    """
    row,col = data.shape
    if percentage_of_max:
        maximum = np.max(data)
        if maximum != 0:
            variance = variance * maximum
        sigma = (variance ** 0.5)
    else:
        sigma = variance ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, (row, col))
    gaussian_noise = gaussian_noise.reshape(row, col)
    noisy_image = data + gaussian_noise
    return noisy_image

def generate_output_data(input_data, exponential_noise, scale, mean, number_of_points_per_direction):
    """
    Generate the E and H waveguide data in each direction with the desired noise added (either exponential or Gaussian)
    :param input_data: array of input mode configurations on which to generate waveguide data
    :param exponential_noise: if True, use exponential noise. If False, use Gaussian noise
    :param scale: the relative amount of noise added; beta for exponential and variance for Gaussian
    :param mean: mean used for Gaussian noise
    :param number_of_points_per_direction: this is the number of cells in the x and y direction of the cross-section
    :return:
    """
    waveguide = WaveguideParameters(number_of_points_per_direction)
    output_data = np.zeros((input_data.shape[0], 12, waveguide.number_of_points_per_direction, waveguide.number_of_points_per_direction))

    for config in range(input_data.shape[0]):
        freq = 1e9
        output_data[config, :, :, :] = generate_e_and_h_data(input_data[config], freq, waveguide)
        for wave in range(12):
            if exponential_noise:
                output_data[config, ][wave, :, :] = add_exponential_noise(output_data[config, ][wave, :, :], scale, True) # Adds noise to the wave based on max for this wave
            else:
                output_data[config, ][wave, :, :] = add_gaussian_noise(output_data[config, ][wave, :, :], scale, mean, True) # Adds noise to the wave based on max for this wave
    return output_data, waveguide