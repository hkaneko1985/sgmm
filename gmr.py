# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
import numpy as np
from scipy.stats import multivariate_normal


def gmr_predict(gmm_model, input_variables, numbers_of_input_variables, numbers_of_output_variables):
    """
    Gaussian Mixture Regression (GMR) based on Gaussian Mixture Model (GMM)
    
    Predict values of variables for forward analysis (regression) and inverse analysis

    Parameters
    ----------
    gmm_model: mixture.gaussian_mixture.GaussianMixture
        GMM model constructed using scikit-learn
    input_variables: numpy.array or pandas.DataFrame
        (autoscaled) m x n matrix of input variables of training data or test data,
        m is the number of sammples and
        n is the number of input variables
        When this is X-variables, it is forward analysis (regression) and
        when this is Y-variables, it is inverse analysis
    numbers_of_input_variables: list
        vector of numbers of input variables
        When this is numbers of X-variables, it is forward analysis (regression) and
        when this is numbers of Y-variables, it is inverse analysis
    numbers_of_output_variables: list
        vector of numbers of output variables
        When this is numbers of Y-variables, it is forward analysis (regression) and
        when this is numbers of X-variables, it is inverse analysis

    Returns
    -------
    mode_of_estimated_mean : numpy.array
        (autoscaled) m x k matrix of output variables estimated using mode of weights,
        k is the number of output variables
    weighted_estimated_mean : numpy.array
        (autoscaled) m x k matrix of output variables estimated using weighted mean,
    estimated_mean_for_all_components : numpy.array
        (autoscaled) l x m x k matrix of output variables estimated for all components,
    weights : numpy.array
        m x l matrix of weights,
    """

    input_variables = np.array(input_variables)

    input_means = gmm_model.means_[:, numbers_of_input_variables]
    output_means = gmm_model.means_[:, numbers_of_output_variables]

    if gmm_model.covariance_type == 'full':
        all_covariances = gmm_model.covariances_
    elif gmm_model.covariance_type == 'diag':
        all_covariances = np.empty(
            [gmm_model.n_components, gmm_model.covariances_.shape[1], gmm_model.covariances_.shape[1]])
        for component_number in range(gmm_model.n_components):
            all_covariances[component_number, :, :] = np.diag(gmm_model.covariances_[component_number, :])
    elif gmm_model.covariance_type == 'tied':
        all_covariances = np.tile(gmm_model.covariances_, (gmm_model.n_components, 1, 1))
    elif gmm_model.covariance_type == 'spherical':
        all_covariances = np.empty([gmm_model.n_components, len(gmm_model.means_), len(gmm_model.means_)])
        for component_number in range(gmm_model.n_components):
            all_covariances[component_number, :, :] = np.diag(
                gmm_model.covariances_[component_number] * np.ones(len(gmm_model.means_)))

    input_output_covariances = all_covariances[:, numbers_of_input_variables, :]
    input_covariances = input_output_covariances[:, :, numbers_of_input_variables]
    input_output_covariances = input_output_covariances[:, :, numbers_of_output_variables]

    # estimated means and weights for all components
    estimated_mean_for_all_components = np.empty(
        [gmm_model.n_components, input_variables.shape[0], len(numbers_of_output_variables)])
    weights = np.empty([gmm_model.n_components, input_variables.shape[0]])
    for component_number in range(gmm_model.n_components):
        estimated_mean_for_all_components[component_number, :, :] = output_means[component_number, :] + (
                    input_variables - input_means[component_number, :]).dot(
            np.linalg.inv(input_covariances[component_number, :, :])).dot(
            input_output_covariances[component_number, :, :])
        weights[component_number, :] = gmm_model.weights_[component_number] * \
                                       multivariate_normal.pdf(input_variables,
                                                               input_means[component_number, :],
                                                               input_covariances[component_number, :, :])

    weights = weights / weights.sum(axis=0)

    # calculate mode of estimated means and weighted estimated means
    mode_of_estimated_mean = np.empty([input_variables.shape[0], len(numbers_of_output_variables)])
    weighted_estimated_mean = np.empty([input_variables.shape[0], len(numbers_of_output_variables)])
    index_for_mode = np.argmax(weights, axis=0)
    for sample_number in range(input_variables.shape[0]):
        mode_of_estimated_mean[sample_number, :] = estimated_mean_for_all_components[index_for_mode[sample_number], sample_number, :]
        weighted_estimated_mean[sample_number, :] = weights[:, sample_number].dot(
            estimated_mean_for_all_components[:, sample_number, :])

    return mode_of_estimated_mean, weighted_estimated_mean, estimated_mean_for_all_components, weights
