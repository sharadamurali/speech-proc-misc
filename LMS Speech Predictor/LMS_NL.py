"""
LMS_NL.py - adaptive LMS algorithm with added nonlinearity
"""
import numpy as np
import math

def LMS_train(train_X, train_Y, N_step, mu, epochs):
    """
    in:
        train_X, train_Y : input and output of the training set
        N_step : Number of samples considered for prediction
        mu : Step size for weight updates
        epochs : number of passes over the entire training data
    
    out:
        error_vec : Array containing the errors after each epoch
    """
    w = np.zeros((N_step-1), dtype = 'f')    # Initializing weight vector
    err_vec = []
    
    for i in range(epochs):
        for j in range(len(train_X)):
            s_unscaled = sigmoid(np.dot(train_X[j], w))   # Predicted output
            s = 2*s_unscaled - 1    # Scaled
            e = train_Y[j] - s   # Error
                
            # Weight update equations
            der = sigmoid(s)*(1 - sigmoid(s))
            w_change = (der * e * train_X[j]) / np.dot(train_X[j], train_X[j])
            w += (mu * w_change)            
            
        err_vec.append(e*e) # Error at the end of the epoch <-- needs to be changed?
        
#        if i == 0:
#            print(w)
    return err_vec

def sigmoid(x):
    return 1 / (1 + math.e**(-x))