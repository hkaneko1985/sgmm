# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of optimization of covariance type and the number of components
# in Gaussian Mixture Regression (GMR), which is supervised Gaussian Mixture Model (GMM)

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
from gmr import gmr_predict
from sklearn import mixture
from sklearn.model_selection import train_test_split

# Settings
max_number_of_components = 20
covariance_types = ['full', 'diag', 'tied', 'spherical']

number_of_all_samples = 500
number_of_test_samples = 200

numbers_of_X = [0, 1, 2]
numbers_of_Y = [3, 4]

# Generate samples for demonstration
np.random.seed(seed=100)
X = np.random.rand(number_of_all_samples, 3) * 10 - 5
y1 = 3 * X[:, 0:1] - 2 * X[:, 1:2] + 0.5 * X[:, 2:3]
y2 = 5 * X[:, 0:1] + 2 * X[:, 1:2] ** 3 - X[:, 2:3] ** 2
y1 = y1 + y1.std(ddof=1) * 0.05 * np.random.randn(number_of_all_samples, 1)
y2 = y2 + y2.std(ddof=1) * 0.05 * np.random.randn(number_of_all_samples, 1)

variables = np.c_[X, y1, y2]
variables_train, variables_test = train_test_split(variables, test_size=number_of_test_samples, random_state=100)

# Standarize X and y
autoscaled_variables_train = (variables_train - variables_train.mean(axis=0)) / variables_train.std(axis=0, ddof=1)
autoscaled_variables_test = (variables_test - variables_train.mean(axis=0)) / variables_train.std(axis=0, ddof=1)

# Grid search using BIC
bic_values = np.empty([max_number_of_components, len(covariance_types)])
for covariance_type_index, covariance_type in enumerate(covariance_types):
    for number_of_components in range(max_number_of_components):
        gmm_model = mixture.GaussianMixture(n_components=number_of_components + 1, covariance_type=covariance_type)
        gmm_model.fit(autoscaled_variables_train)
        bic_values[number_of_components, covariance_type_index] = gmm_model.bic(autoscaled_variables_train)

# plot
plt.rcParams["font.size"] = 18
plt.figure()
plt.plot(bic_values[:, 3], 'r-', label='spherical')
plt.plot(bic_values[:, 2], 'k-', label='tied')
plt.plot(bic_values[:, 1], 'g-', label='diag')
plt.plot(bic_values[:, 0], 'b-', label='full')
plt.xlabel('Number of components')
plt.ylabel('BIC values')
plt.legend(bbox_to_anchor=(1.05, 0.5, 0.5, .100),
           borderaxespad=0., )
plt.show()

# Optimal parameters
optimal_index = np.where(bic_values == bic_values.min())
optimal_number_of_components = optimal_index[0][0] + 1
optimal_covariance_type = covariance_types[optimal_index[1][0]]

# GMM
gmm_model = mixture.GaussianMixture(n_components=optimal_number_of_components, covariance_type=optimal_covariance_type)
gmm_model.fit(autoscaled_variables_train)

# Forward analysis (regression)
mode_of_estimated_mean_of_Y, weighted_estimated_mean_of_Y, estimated_mean_of_Y_for_all_components, weights_for_X = \
    gmr_predict(gmm_model, autoscaled_variables_test[:, numbers_of_X], numbers_of_X, numbers_of_Y)

# Inverse analysis
mode_of_estimated_mean_of_X, weighted_estimated_mean_of_X, estimated_mean_of_X_for_all_components, weights_for_Y = \
    gmr_predict(gmm_model, autoscaled_variables_test[:, numbers_of_Y], numbers_of_Y, numbers_of_X)

# Check results of forward analysis (regression)
print('Results of forward analysis (regression)')
predicted_ytest_all = mode_of_estimated_mean_of_Y
plt.rcParams["font.size"] = 18
for Y_number in range(len(numbers_of_Y)):
    predicted_ytest = np.ndarray.flatten(predicted_ytest_all[:, Y_number])
    predicted_ytest = predicted_ytest * variables_train[:, numbers_of_Y[Y_number]].std(ddof=1) + \
                      variables_train[:, numbers_of_Y[Y_number]].mean()
    # yy-plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(variables_test[:, numbers_of_Y[Y_number]], predicted_ytest)
    YMax = np.max(np.array([np.array(variables_test[:, numbers_of_Y[Y_number]]), predicted_ytest]))
    YMin = np.min(np.array([np.array(variables_test[:, numbers_of_Y[Y_number]]), predicted_ytest]))
    plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
             [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
    plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlabel("Actual Y")
    plt.ylabel("Estimated Y")
    plt.show()
    # r2p, RMSEp, MAEp
    print("r2p: {0}".format(float(1 - sum((variables_test[:, numbers_of_Y[Y_number]] - predicted_ytest) ** 2) / sum(
        (variables_test[:, numbers_of_Y[Y_number]] - variables_train[:, numbers_of_Y[Y_number]].mean()) ** 2))))
    print("RMSEp: {0}".format(float((sum((variables_test[:, numbers_of_Y[Y_number]] - predicted_ytest) ** 2) / len(
        variables_test[:, numbers_of_Y[Y_number]])) ** 0.5)))
    print("MAEp: {0}".format(float(sum(abs(variables_test[:, numbers_of_Y[Y_number]] - predicted_ytest)) / len(
        variables_test[:, numbers_of_Y[Y_number]]))))

# Check results of inverse analysis
print('---------------------------')
print('Results of inverse analysis')
estimated_X_test = mode_of_estimated_mean_of_X * variables_train[:, numbers_of_X].std(ddof=1) + \
                   variables_train[:, numbers_of_X].mean()
calculated_Y_from_estimated_X_test = np.empty([number_of_test_samples, 2])
calculated_Y_from_estimated_X_test[:, 0:1] = 3 * estimated_X_test[:, 0:1] - 2 * estimated_X_test[:,
                                                                                1:2] + 0.5 * estimated_X_test[:, 2:3]
calculated_Y_from_estimated_X_test[:, 1:2] = 5 * estimated_X_test[:, 0:1] + 2 * estimated_X_test[:,
                                                                                1:2] ** 3 - estimated_X_test[:,
                                                                                            2:3] ** 2
for Y_number in range(len(numbers_of_Y)):
    predicted_ytest = np.ndarray.flatten(calculated_Y_from_estimated_X_test[:, Y_number])
    # yy-plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(variables_test[:, numbers_of_Y[Y_number]], predicted_ytest)
    YMax = np.max(np.array([np.array(variables_test[:, numbers_of_Y[Y_number]]), predicted_ytest]))
    YMin = np.min(np.array([np.array(variables_test[:, numbers_of_Y[Y_number]]), predicted_ytest]))
    plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
             [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
    plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
    plt.xlabel("Actual Y")
    plt.ylabel("Estimated Y")
    plt.show()
    # r2p, RMSEp, MAEp
    print("r2p: {0}".format(float(1 - sum((variables_test[:, numbers_of_Y[Y_number]] - predicted_ytest) ** 2) / sum(
        (variables_test[:, numbers_of_Y[Y_number]] - variables_train[:, numbers_of_Y[Y_number]].mean()) ** 2))))
    print("RMSEp: {0}".format(float((sum((variables_test[:, numbers_of_Y[Y_number]] - predicted_ytest) ** 2) / len(
        variables_test[:, numbers_of_Y[Y_number]])) ** 0.5)))
    print("MAEp: {0}".format(float(sum(abs(variables_test[:, numbers_of_Y[Y_number]] - predicted_ytest)) / len(
        variables_test[:, numbers_of_Y[Y_number]]))))
