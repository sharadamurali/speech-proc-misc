# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:36:37 2018

@author: sharada

GMM-EM algorithm
1. Generate gaussian data and concatenate into an array
2. Initialize random values for mu, sigma/variance, weights
3. E-step: Calculate probability of each data point as generated from the different gaussians
4. M-step: Adjust estimated mu, variance, weights such that log likelihood is maximized
"""

import numpy as np
import matplotlib.pyplot as plt

GENERATE_DATA = 1
DISPLAY = 0
GENERATE_PROB = 1

def plot_histogram(data, labels):
    """
    Each data point is associated with a label signifying the gaussian which produced it
    """
    lab_list = np.unique(labels)
    
    plt.figure()
    plt.title('Data Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # For each label, collect data and plot it as a histogram
    for i in range(len(lab_list)):
        curr_gauss = np.array([])
        
        for j in range(data.size):
            if gauss[j] == lab_list[i]:
                curr_gauss = np.append(curr_gauss, data[j])
        
        plt.hist(curr_gauss, bins=100)


# Returns the probability of a data point coming from the distribution
def prob(x, w, mu, var):
    # Mu and sigma are vectors
    exponent = ((x-mu)**2)/(2*var)
    prob_x = (w/np.sqrt(2*np.pi*var)) * np.exp(-exponent)
    return prob_x


if __name__ == '__main__':
    
    np.random.seed(0)   # For reproducibility; can be commented out.
    
    if GENERATE_DATA:
        # Parameters for data
        mu = [-1, 0.7, 3, 4.5]
        var = [0.15, 0.1, 0.2, 0.3]
        
        n = len(mu)   # Number of gaussians
        n_i = 250   # Number of data points from a single gaussian
        
        # Generating data
        data = np.array([])
        
        for i in range(n):        
            data_i = np.random.normal(mu[i], np.sqrt(var[i]), n_i)        
            data = np.append(data, data_i)
    
    
    
    eps = 0.001 # Stopping criterion   
    
    # Initializing gaussians
    
    # Pick two random points from the data for random mu
    mu_pred = data[np.random.randint(0, len(data), size=n)]
    # Initialize variance as non-negative values
    var_pred = np.random.rand(1,n)[0]
    # Initialize weights; non-negative and sum to 1
    w_pred = np.random.rand(1, n)[0]
    w_pred /= np.sum(w_pred)
    
    print('Initial values of mean, variance and mixture weights:')
    print(mu_pred)
    print(var_pred)
    print(w_pred)
    
    # Assigning points to different gaussians
    gauss = np.random.randint(0, 3, (data.size,1))
    plot_histogram(data, gauss)
    # Probability/ responsibility matrix
    prob_mat = np.zeros((data.size, n))
    ll_mat = np.copy(prob_mat)
    
    err = 1 # Maximum error while updating means
    num_iter = 0    # Keep track of the number of iterations
    
    # Log likelihoods
    log_l = np.array([])
    
    while err > eps:    # Repeat until convergence
        # Expectation step -- assigning ownership weights to each of the gaussians
        for j in range(data.size):
            prob_mat[j] = prob(data[j], w_pred, mu_pred, var_pred)
            ll_mat[j] = prob_mat[j]
            prob_mat[j] /= np.sum(prob_mat[j])  # Normalizing for probability distribution
            gauss[j] = np.argmax(prob_mat[j])   # New assignation -- for plotting
        
        # Maximization step -- adjusting parameters
        # Adjusting mu
        weights = np.multiply(data.T, prob_mat.T).T
        mu_new = np.sum(weights, axis = 0)/np.sum(prob_mat, axis = 0)
        
        # Calculating error:
        err = np.max(np.abs(mu_pred - mu_new))
        # Update mean
        mu_pred = mu_new
        
        # Adjusting weights
        w_pred = np.sum(prob_mat, axis=0)/data.size
        
        # Adjusting variance 
        
        for j in range(n):
            var_sum = 0
            for k in range(data.size):
                var_sum += prob_mat[k, j] * ((data[k] - mu_pred[j])**2)
            
            var_pred[j] = var_sum / np.sum(prob_mat[:,j])
        
        
        # Calculate log likelihood
        curr_ll = np.sum(np.log(np.sum(ll_mat, axis = 1)))
        log_l = np.append(log_l, curr_ll)
        
        # Display updated results
        if (num_iter % 10 == 0) and DISPLAY:        
            print()
            print('Iteration ', num_iter)
            print('Weights: ', w_pred)       
            print('Mean: ', mu_pred)
            print('Variance: ', var_pred)
            
            plot_histogram(data, gauss)
        
        num_iter += 1   # Increase counter
    
    # Display final results
    print()
    print('# Iterations to convergence: ', num_iter)
    print('Final Weights: ', w_pred)       
    print('Mean: ', mu_pred)
    print('Variance: ', var_pred)    
    plot_histogram(data, gauss)
    
    # Plot the log likelihood
    plt.figure()
    plt.title('Log-likelihood variation')
    plt.xlabel('# Iterations')
    plt.ylabel('Log-likelihood')
    plt.plot(log_l)
    
    """
    Gives the probability distribution of a data point belonging to the "learned
    gaussians"
    """
    if GENERATE_PROB:  
        print()
        val = input('Enter point value: ')
        val = float(val)
        prob_val = prob(val, w_pred, mu_pred, var_pred)
        prob_val /= np.sum(prob_val)
        print('Probabilities: ', prob_val)