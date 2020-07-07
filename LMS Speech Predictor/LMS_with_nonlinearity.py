"""
Main file for the LMS predictor with non-linearity

List of youtube files to be downloaded is specified here:
    speech_files : List of links to be downloaded with "clean" speech
    noise_file : Noise file to be mixed with clean speech

Other files used in this program:
    ExtractAudio : returns numpy array with RAW values (scaled)
    LMS : Performs LMS training and returns final errors and optimal weights
    

In this program, only one speech file and one noise file are used.
"""
import numpy as np
import matplotlib.pyplot as plt

import ExtractAudio
import LMS_NL

def form_train_set(raw_data_0, N, num_samples):
    raw_data = np.array(raw_data_0, dtype = 'f')
    
    # Normalizing
    raw_data /= np.abs(np.max(raw_data))
    
    # Reshaping training data to (x,y) pairs
    train_X = np.empty((num_samples-1, N-1))
    train_Y = np.empty((num_samples-1))
    
    for i in range(num_samples-1):
        train_X[i] = raw_data[i:(i+N-1)]
    train_Y = raw_data[N:N+num_samples-1]
    
    return train_X, train_Y
    

if __name__ == "__main__":
    
    # List of file URLs to be downloaded
    speech_files = ['https://www.youtube.com/watch?v=arj7oStGLkU&t=75s'];
    noise_file = ['https://www.youtube.com/watch?v=HbVYuPogyP0'];
    
    # LMS Parameters
    N = 100 # for the N-step predictor
    num_samples = [40000, 50000] # Different training set sizes
    epoch = 20
    mu = 0.001  # Step size for the weight update
    
    """
    Part 1: Clean Speech only
    """
    print("Part 1: Training the N-step LMS predictor on Clean Speech", end = '\n')
    # Get RAW data for the training set
    speech_file_name = ExtractAudio.get_audio(speech_files, 'speech', '14')
    raw_data_speech = ExtractAudio.get_RAW(speech_file_name) #"new_trim_fin.wav")#
    
    # Initialize empty error vector for plotting convergence
    err_vec = np.empty((len(num_samples), epoch))
    
    for n in range(len(num_samples)):
        # Form the training set
        train_X, train_Y = form_train_set(raw_data_speech, N, num_samples[n])
        # Run the LMS predictor (train)
        print('Running the LMS predictor...')
        err_vec[n] = LMS_NL.LMS_train(train_X, train_Y, N, mu, epoch)
        print('Done.')
    
    # Plotting results
    
    x_ax = list(range(1, epoch+1))
    
    print('Plotting results')
    plt.figure(1)
    plt.title('Variation in error with audio segment length')
    plt.xlabel("Epochs")
    plt.ylabel("Error")    
    
    for i in range(len(num_samples)):
        plt.plot(x_ax, err_vec[i])
    
    plt.legend(num_samples)
    plt.show()
    
    """
    Part 2: Mixing speech with noise
    """
    print('Part 2: LMS predictor on noisy (corrupted) speech')
    # Download the noise file and get mixed data
    noise_file_name = ExtractAudio.get_audio(noise_file, 'noise', '0')
    mixed_file_name = ExtractAudio.mix_audio(speech_file_name, noise_file_name)
    raw_data_noisy = ExtractAudio.get_RAW(mixed_file_name)
    
    # Initialize empty error vector for plotting convergence
    err_vec_noisy = np.empty(epoch)
    
    # Form the training set
    train_X_noisy, train_Y_noisy = form_train_set(raw_data_noisy, N, num_samples[0])
    # Run the LMS predictor (train)
    print('Running the LMS predictor...')
    err_vec_noisy = LMS_NL.LMS_train(train_X_noisy, train_Y_noisy, N, mu, epoch)
    print('Done')
    
    # Plotting results
        
    print('Plotting results')
    
    plt.figure(2)
    plt.title('Effect of noise on prediction error')
    plt.xlabel("Epochs")
    plt.ylabel("Error")  
    
    plt.plot(x_ax, err_vec[0], x_ax, err_vec_noisy)
    plt.legend(('Clean Speech', 'Noisy Speech'))
    plt.show()