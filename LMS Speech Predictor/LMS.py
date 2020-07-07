"""
LMS.py - adaptive LMS algorithm for an N-step predictor
"""
import numpy as np

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
            s = np.dot(train_X[j], w)   # Predicted output
            e = train_Y[j] - s   # Error
                
            # Weight update equations
            w_change = (e * train_X[j]) / np.dot(train_X[j], train_X[j])
            w += (mu * w_change)            
            
        err_vec.append(e*e) # Error at the end of the epoch <-- needs to be changed?
        
#        if i == 0:
#            print(w)
    return err_vec
