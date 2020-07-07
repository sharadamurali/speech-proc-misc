# -*- coding: utf-8 -*-
"""
Keras train and test on the input data
"""


from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.initializers import glorot_normal
from keras import optimizers
from sklearn.model_selection import train_test_split
from time import gmtime, strftime

import numpy as np
import matplotlib.pyplot as plt

def train_denoiser(X, Y):
    
    # Split into train and validation sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=73)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=2373)
    
    n_input_dim = X_train.shape[1]
    n_output_dim = Y_train.shape[1]

    n_hidden = 3000
    n_encoder = 128

    input_shape = X_train[0,:].shape
    
    # Defining the layers and adding Batch Normalization and Dropout
    InputLayer = Input(shape = input_shape, name  = 'input_layer')
    InputLayer2 = BatchNormalization(axis=1, momentum=0.6)(InputLayer)
    InputLayer3 = Dropout(0.2)(InputLayer2)
    
    FirstLayer = Dense(n_hidden, activation = 'relu', name = 'layer_1', kernel_initializer=glorot_normal(seed=0))(InputLayer3)
    FirstLayer2 = BatchNormalization(axis=1, momentum=0.6)(FirstLayer)
    FirstLayer3 = Dropout(0.2)(FirstLayer2)
    
    EncoderLayer = Dense(n_encoder, activation = 'relu', name = 'layer_2', kernel_initializer=glorot_normal(seed=0))(FirstLayer3)
    #EncoderLayer2 = BatchNormalization(axis=1, momentum=0.6)(EncoderLayer)
    #EncoderLayer3 = Dropout(0.2)(EncoderLayer2)
    
    ThirdLayer = Dense(n_hidden, activation = 'relu', name = 'layer_3', kernel_initializer=glorot_normal(seed=0))(EncoderLayer)
    ThirdLayer2 = BatchNormalization(axis=1, momentum=0.6)(ThirdLayer)
    ThirdLayer3 = Dropout(0.2)(ThirdLayer2)
    
    OutputLayer = Dense(n_output_dim, activation = 'sigmoid', name = 'output_layer', kernel_initializer=glorot_normal(seed=0))(ThirdLayer3)
    
    # Building the model
    model = Model(inputs = [InputLayer], outputs = [OutputLayer])
    
#    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.001, amsgrad=False)
#    model.compile(loss='mse',optimizer=opt)

    #plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)
    
    # Compiling the model
    model.compile(optimizer = 'adam', loss = 'cosine', metrics = ['accuracy'])
    model.summary()
    
    tensorboard = TensorBoard(log_dir="logs/"+strftime("%Y_%m_%d-%H_%M_%S", gmtime()) , histogram_freq=0, batch_size=320, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None  )
    
    # Training the Model
        
    hist = model.fit(X_train, Y_train, batch_size = 256, epochs=500, verbose=1, validation_data=([X_val], [Y_val]), callbacks=[tensorboard])
    
    plt.figure(1)
    plt.plot(hist.history['loss'], label='Loss')
    plt.plot(hist.history['acc'], label='Accuracy')
    plt.legend(loc='best')
    plt.title('Training Accuracy and Loss')
    plt.show()
    
    results = model.evaluate(X_test, Y_test, batch_size=len(Y_test))
    #print("Test Loss: %.3f" %results)
    print("Test Loss: %.3f\nTest Accuracy: %.3f" %(results[0], results[1]))
    
    return model



def denoise_signal(model_name, test_data):
    denoiser = load_model(model_name)
    
    Y_pred = denoiser.predict(test_data)
#    gains = np.abs(Y_pred)/np.abs(freq_sig_noisy)
#    
#    X_pred = np.multiply(gains, freq_sig_noisy)
    
    return Y_pred
    