# -*- coding: utf-8 -*-
"""
@author: Sharada
GMM fitting using a neural network
"""
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model, load_model
from keras.initializers import glorot_normal
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

TRAIN = 1
TEST = 1

GENERATE_DATA = 1
PLOT = 0

# Creates a model for training
def create_model(X, Y):
    try:
        input_shape = X.shape[1]
    except:
        input_shape = 1
    output_shape = Y.shape[1]

    n_hidden = 200   
    
    # Defining the layers and adding Batch Normalization and Dropout
    InputLayer = Input(shape = (input_shape,), name  = 'input_layer')
    InputLayer2 = BatchNormalization(axis=1, momentum=0.6)(InputLayer)
    InputLayer3 = Dropout(0.2)(InputLayer2)
    
    FirstLayer = Dense(n_hidden, activation = 'relu', name = 'layer_1', kernel_initializer=glorot_normal(seed=0))(InputLayer3)
    FirstLayer2 = BatchNormalization(axis=1, momentum=0.6)(FirstLayer)
    FirstLayer3 = Dropout(0.2)(FirstLayer2)
    
    SecondLayer = Dense(n_hidden, activation = 'relu', name = 'layer_2', kernel_initializer=glorot_normal(seed=0))(FirstLayer3)
    SecondLayer2 = BatchNormalization(axis=1, momentum=0.6)(SecondLayer)
    SecondLayer3 = Dropout(0.2)(SecondLayer2)
    
    ThirdLayer = Dense(n_hidden, activation = 'relu', name = 'layer_3', kernel_initializer=glorot_normal(seed=0))(SecondLayer3)
    ThirdLayer2 = BatchNormalization(axis=1, momentum=0.6)(ThirdLayer)
    ThirdLayer3 = Dropout(0.2)(ThirdLayer2)
    
    OutputLayer = Dense(output_shape, activation = 'softmax', name = 'output_layer', kernel_initializer=glorot_normal(seed=0))(ThirdLayer3)
    
    # Building the model
    model = Model(inputs = [InputLayer], outputs = [OutputLayer])
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    
    return model
    
def train(model, X, Y):
    
    # Split into train and validation sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=73)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=2373)   
        
    # Training the Model
    hist = model.fit(X_train, Y_train, epochs = 200, verbose = 0, validation_data=([X_val], [Y_val]))
    
    if PLOT:
        plt.figure()
        plt.plot(hist.history['loss'], label='Training_Loss')
        plt.plot(hist.history['acc'], label='Training_Accuracy')
        plt.plot(hist.history['val_loss'], label='Validation_Loss')
        plt.plot(hist.history['val_acc'], label='Validation_Accuracy')
        plt.legend(loc='best')
        plt.title('Training, Validation Accuracy and Loss')
        plt.show()
    
    results = model.evaluate(X_test, Y_test, batch_size=len(Y_test))
    #print("Test Loss: %.3f" %results)
    print("Test Loss: %.3f\nTest Accuracy: %.3f" %(results[0], results[1]))
    
    return model

def predict_prob(model_name, test_data):
    pred_model = load_model(model_name)
    # Gives the softmax output of a value
    Y_pred = pred_model.predict(test_data)
    # Y_pred = np.round(Y_pred)
    
    return Y_pred

if __name__ == '__main__':
        
    np.random.seed(0)   # For reproducibility; can be commented out.
    
    if GENERATE_DATA:
        # Parameters for data
        mu = [-1, 0.7, 3, 4.5]
        var = [0.15, 0.1, 0.2, 0.3]
        
        n = len(mu)   # Number of gaussians
        n_i = 1000   # Number of data points from a single gaussian
        
        # Generating data
        data = np.array([])
        # Form Y label to train the NN        
        label = np.zeros((0, n))
        
        for i in range(n):        
            data_i = np.random.normal(mu[i], np.sqrt(var[i]), n_i)        
            data = np.append(data, data_i)
            
            for j in range(data_i.size):
                curr_label = np.zeros((1,n))
                curr_label[:, i] = 1
                label = np.append(label, curr_label, axis = 0)
    
    if TRAIN:
        # X_train = data
        # Y_train = label
        
        model = create_model(data, label)
        model = train(model, data, label)
        # Save the model
        model.save('gmmfit.h5')
    
    if TEST:
        val = input('Enter point value: ')
        val = float(val)
        prob_val = predict_prob('gmmfit.h5', [val])
        print('The probability distribution is: ', prob_val)
        