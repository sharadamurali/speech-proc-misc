"""
Main function for the autoencoder as a denoiser
"""

from keras.models import Model, load_model
import numpy as np
import numpy.fft as fft
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

import ExtractAudio
import Denoiser




"""
Global Variables
"""
DOWNLOAD = 1            # Download the audio data
MIX = 1                 # Mix noise
PLOT_WAVEFORM = 1       # Plot clean, noisy and denoised waveforms
TRAIN = 1               # Train the denoiser
TEST = 1                # Test the denoiser
TEST_RECONSTRUCTION = 0 # Test reconstructing the signal after FFT, mel conversions




"""
Functions
"""
# Mixing noise with the speech signal
def mix_noise(S, N, w):
    X = S + w*N
    return X



# Time domain to frequency domain
def time_to_freq(time_data, window_size, win_shift):
    window = np.hamming(window_size)
    freq_sig = [fft.rfft(window*time_data[0:window_size])] #np.zeros((num_rows, int(window_size/2 + 1)))
    
    start = win_shift
    end = start + window_size
    
    while end < len(time_data):        
        curr_freq_sig = fft.rfft(window*time_data[start:end])
        freq_sig = np.append(freq_sig, [curr_freq_sig], axis = 0)
        start += win_shift
        end = start + window_size
    
    return freq_sig



# Back to time domain
def freq_to_time(freq_data, window_size, win_shift):
    
    time_len = (len(freq_data) * win_shift) + (2 * window_size)
    time_sig = np.zeros(time_len)
    
    start = 0
    
    for i in range(len(freq_data)):    
        end = start + window_size
        curr_time = fft.irfft(freq_data[i])
        time_sig[start:end] += curr_time
        start += window_shift
    
    time_sig = ExtractAudio.normalize_audio(time_sig)
    
    return time_sig



"""
Generating the freq2mel (and conversely, mel2freq) matrix..
"""
def mel(freq):
    # Returns the mel value for a particular frequency value
    mel_val = 2595 * np.log10(1 + (freq/700)) # Check this eq!
    return mel_val

def freq(mel):
    # Returns the frequency value for a particular mel value
    freq_val = 700 * ((10 ** (mel / 2595)) - 1) # Check this eq!
    return freq_val


    
def construct_freq_to_mel(samp_rate, bins):
    """
    Mel matrix for every filterbank
    """
    low_bound = mel(0)
    high_bound = mel(samp_rate/2)
    
    step = (high_bound - low_bound)/(bins+1)
    
    freq_bin_values = np.zeros((bins+2))
    
    i = low_bound
    k = 0
    
    while i < high_bound:
        freq_bin_values[k] = freq(i) # Given the mel value, get the frequency value
        i = i + step
        k = k + 1
    freq_bin_values[k] = freq(high_bound)
    
    # Converting the frequencies to the FFT bin
    freq_bin_values = np.floor((window_size + 2) * freq_bin_values / samp_rate)
    
    freq2mel = np.zeros((window_size // 2 + 1, bins))
    
    """
    Construct the mel filterbanks - each filterbank is each column of the matrix.
    When the row index is less than the p
    """
    freq_idx = 1
    
    for col in range(bins): #Column
        for row in range(len(freq2mel)):  # Rows
            if (row >= freq_bin_values[freq_idx-1]) and (row <= freq_bin_values[freq_idx]):
                freq2mel[row, col] = (row - freq_bin_values[freq_idx-1]) / (freq_bin_values[freq_idx] - freq_bin_values[freq_idx-1])
            elif (row > freq_bin_values[freq_idx]) and (row <= freq_bin_values[freq_idx+1]):
                freq2mel[row, col] = (freq_bin_values[freq_idx+1] - row) / (freq_bin_values[freq_idx+1] - freq_bin_values[freq_idx])
        freq_idx += 1
    
    # Normalize each column (divide by the sum of the column)
    sum_array = np.sum(freq2mel, axis = 0)
    freq2mel_2 = np.divide(freq2mel, sum_array)
    freq2mel_2 = np.nan_to_num(freq2mel_2)  # Fixing NaNs
    
    return freq2mel_2



def construct_mel_to_freq(freq2mel):
    mel2freq = np.transpose(freq2mel)
    # Normalizing
    sum_array = np.sum(mel2freq, axis = 0)
    mel2freq_2 = np.divide(mel2freq, sum_array)
    mel2freq_2 = np.nan_to_num(mel2freq_2)
    
    return mel2freq_2





"""
Main function
"""
if __name__ == "__main__":    
    
    # URLs for downloading the files -- find data!
    speech_files = [['https://www.youtube.com/watch?v=NAxIhG8jDWI'], 
                   ['https://www.youtube.com/watch?v=18uDutylDa4'], 
                   ['https://www.youtube.com/watch?v=arj7oStGLkU&t=75s']]
    noise_file = ['https://www.youtube.com/watch?v=HbVYuPogyP0']
    
    # Windowing parameters
    window_size = 1024
    window_shift = 256
    
    samp_rate = 16000
    bins = 40
    
    """
    Downloading speech and noise files and mixing them
    """
    #speech_file_name = ['speech0_trim', 'speech1_trim', 'speech2_trim']
    #noise_file_name = 'noise_trim'
    
    speech_file_length = 7
    
    if DOWNLOAD: 
        speech_file_name = []
        for i in range(len(speech_files)):
            speech_file_name.append(ExtractAudio.get_audio(speech_files[i], 'speech' + str(i), '60', str(speech_file_length)))
        
        noise_file_name = ExtractAudio.get_audio(noise_file, 'noise', '0', str(speech_file_length*3))
    
    
    # Get RAW data for the training set - speech and noise
    raw_data_speech = np.array([])
    for i in range(len(speech_files)):
        raw_data_speech = np.append(raw_data_speech, ExtractAudio.get_RAW(speech_file_name[i]))
    raw_data_noise = ExtractAudio.get_RAW(noise_file_name)
    
    if MIX:
        print('Mixing Clean Speech and noise.\n\n')
        noisy_signal = mix_noise(raw_data_speech, raw_data_noise, 0.75)
        noisy_signal = ExtractAudio.normalize_audio(noisy_signal)
        wavfile.write('noisy_signal.wav', samp_rate, noisy_signal)
    
    # FFT of Clean Speech
    freq_sig_clean = time_to_freq(raw_data_speech, window_size, window_shift)
    # Taking the FFT of the noisy signal
    freq_sig_noisy = time_to_freq(noisy_signal, window_size, window_shift)
    
    # Constructing Freq Components of Y_Train
    #freq_Y_train = np.divide(freq_sig_clean, freq_sig_noisy)
    
    # Constructing freq2mel and mel2freq matrices
    freq2mel = construct_freq_to_mel(samp_rate, bins)    
    mel2freq = construct_mel_to_freq(freq2mel)
    
    # Getting the mel coefficients for X and Y of train set
    mel_noisy = np.dot(np.abs(freq_sig_noisy), freq2mel)    # X_Train
    mel_Y = np.dot(np.abs(freq_sig_clean), freq2mel)  # Y_Train
    #mel_clean = np.dot(np.abs(freq_sig_clean), freq2mel)  
    #mel_Y = np.divide(mel_clean, mel_noisy) # Y_Train
    
    if TEST_RECONSTRUCTION:
        #Converting back to the frequency domain
        freq_sig_test = np.dot(mel_noisy, mel2freq)  
        
        # Reconstructing with phase
        freq_sig_test = np.multiply(freq_sig_test, freq_sig_clean)    
        time_sig = freq_to_time(freq_sig_test, window_size, window_shift)
        
        # Write the data to a WAV file
        wavfile.write('test_noisy.wav', samp_rate, time_sig)
    
    """
    Form training and test data
    """
    
    end_idx = int(0.8*len(mel_noisy))   # Taking 80% of the clip for training
    
    X_train = mel_noisy[:end_idx, :]
    X_test = mel_noisy[end_idx:, :]
    Y_train = mel_Y[:end_idx, :]
    Y_test = mel_Y[end_idx:, :]
    
    freq_test = freq_sig_noisy[end_idx:,:]
    
    if TRAIN:
        print('Training the model...')
        model = Denoiser.train_denoiser(X_train, Y_train)
        
        # Save the model
        model.save('denoiser.h5')
    
    if TEST:
        print('Testing the model...')
        mel_pred = Denoiser.denoise_signal('denoiser.h5', X_test)
        
        # Convert mel to frequency
        freq_pred = np.dot(mel_pred, mel2freq)
        
        # Reconstructing with phase
        freq_pred_phase = np.multiply(freq_pred, freq_test)
        time_pred = freq_to_time(freq_pred_phase, window_size, window_shift)
        
        # Write the data to a .WAV file
        print('Writing the denoised signal to a .wav file')
        wavfile.write('denoised_signal.wav', samp_rate, time_pred)
    
    if PLOT_WAVEFORM:
        pred_sig_len = len(time_pred)
        f, xarr = plt.subplots(2, sharex = True)
        f.suptitle('Comparison of Original and Noisy Waveforms')
        xarr[0].plot(raw_data_speech)
        xarr[1].plot(noisy_signal)
        #xarr[2].plot(time_pred[:20000])
        
        plt.figure(3)
        plt.title('Waveform of denoised signal')
        plt.plot(time_pred)