# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""
import numpy as np
import numpy.matlib
from scipy.stats import multivariate_normal
import math
from sklearn import mixture

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
    if input_variables.ndim == 0:
        input_variables = np.reshape(input_variables, (1, 1))
    elif input_variables.ndim == 1:
        input_variables = np.reshape(input_variables, (1, input_variables.shape[0]))

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
            
    if all_covariances.shape[2] == len(numbers_of_input_variables) + len(numbers_of_output_variables):
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
    else:
        mode_of_estimated_mean = np.ones([input_variables.shape[0], len(numbers_of_output_variables)]) * -99999
        weighted_estimated_mean = np.ones([input_variables.shape[0], len(numbers_of_output_variables)]) * -99999
        weights = np.zeros([gmm_model.n_components, input_variables.shape[0]])
        estimated_mean_for_all_components = np.zeros(
            [gmm_model.n_components, input_variables.shape[0], len(numbers_of_output_variables)])
        
    return mode_of_estimated_mean, weighted_estimated_mean, estimated_mean_for_all_components, weights

def gmr_cvopt(dataset, numbers_of_input_variables, numbers_of_output_variables, covariance_types, max_number_of_components, fold_number):
    """
    Hyperparameter optimization for Gaussian Mixture Regression (GMR)

    Parameters
    ----------
    dataset: numpy.array or pandas.DataFrame
        m x n matrix of datasetvariables of training data or test data,
        m is the number of sammples and
        n is the number of both input and output variables
    numbers_of_input_variables: list
        vector of numbers of input variables
        When this is numbers of X-variables, it is forward analysis (regression) and
        when this is numbers of Y-variables, it is inverse analysis
    numbers_of_output_variables: list
        vector of numbers of output variables
        When this is numbers of Y-variables, it is forward analysis (regression) and
        when this is numbers of X-variables, it is inverse analysis
    covariance_types: list
        candidates of covariance types such as ['full', 'diag', 'tied', 'spherical']
    max_number_of_components: int
        number of maximum components in GMM
    fold_number: int
        number of fold in cross-validation        

    Returns
    -------
    best_covariance_type : str
        best covariance type
    best_number_of_components : int
        best number of components
    """
    
    dataset = np.array(dataset)
    autoscaled_dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0, ddof=1)

    r2cvs = []
    for covariance_type in covariance_types:
        for number_of_components in range(max_number_of_components):
            estimated_y_in_cv = np.zeros([dataset.shape[0], len(numbers_of_output_variables)])
            
            min_number = math.floor(dataset.shape[0] / fold_number)
            mod_number = dataset.shape[0] - min_number * fold_number
            index = np.matlib.repmat(np.arange(1, fold_number+1, 1), 1, min_number).ravel()
            if mod_number != 0:
                index = np.r_[index, np.arange( 1, mod_number+1, 1 )]
#            np.random.seed(999) 
            fold_index_in_cv = np.random.permutation(index)
            np.random.seed()
            for fold_number_in_cv in np.arange(1, fold_number+1, 1):
                dataset_train_in_cv = autoscaled_dataset[fold_index_in_cv != fold_number_in_cv, :]
                dataset_test_in_cv = autoscaled_dataset[fold_index_in_cv == fold_number_in_cv, :]
                gmm_model = mixture.GaussianMixture(n_components=number_of_components + 1, covariance_type=covariance_type)
                gmm_model.fit(dataset_train_in_cv)
                
                mode_of_estimated_mean_of_Y, weighted_estimated_mean_of_Y, estimated_mean_of_Y_for_all_components, weights_for_X = \
                    gmr_predict(gmm_model, dataset_test_in_cv[:, numbers_of_input_variables], numbers_of_input_variables, numbers_of_output_variables)
            
                estimated_y_in_cv[fold_index_in_cv == fold_number_in_cv, :] = mode_of_estimated_mean_of_Y # 格納
            
            y = np.ravel(autoscaled_dataset[:, numbers_of_output_variables])
            y_pred = np.ravel(estimated_y_in_cv)
            r2 = float(1 - sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2))
            r2cvs.append(r2)
    max_r2cv_number = np.where(r2cvs == np.max(r2cvs))[0][0]
    best_covariance_type = covariance_types[max_r2cv_number // max_number_of_components]
    best_number_of_components = max_r2cv_number % max_number_of_components + 1
    
    return best_covariance_type, best_number_of_components
    
    