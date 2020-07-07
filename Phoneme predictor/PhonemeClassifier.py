"""
Phoneme classifier
"""

from keras.models import Model, load_model
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import librosa

#import ExtractAudio
import KerasPredictor




"""
Global Variables
"""
TRAIN = 1               # Train the denoiser
TEST = 1              # Test the denoiser

"""
Functions
"""
# Returns MFCC array for the given speech file (with directory, if applicable)
def get_MFCC(filename, num_mfcc, window_size, window_shift):
    # Get RAW data
    rate, raw_data_speech = wavfile.read(filename)
    
    # Get MFCCs from librosa
    freq_sig = librosa.core.stft(raw_data_speech.astype(float), n_fft = window_size, hop_length = window_shift, center = False)
    mel_spec = librosa.feature.melspectrogram(sr = rate, S = freq_sig)
    mfcc = librosa.feature.mfcc(S = librosa.power_to_db(mel_spec), sr = rate, n_mfcc = num_mfcc).T  
    
    return mfcc

# Returns list of phonemes for a particular file
def get_phonemes_data(filename):
    # Get line for the current speech file from the text file
    with open('all_alignments.txt', 'r') as infile:
        for line in infile:
            phonemes  = line.split()
            
            if filename == phonemes[0]:
                print('File found in phoneme list.\n')
                break;
    
    # List of phonemes for the current file
    phonemes = phonemes[1:]
    
    # Removing extra characters
    for i in range(len(phonemes)):
        phonemes[i] = phonemes[i].replace("_B", "").replace("_I", "").replace("_E", "").replace("_S", "")
    
    return phonemes
        
# Returns one-hot vector for each phoneme
def phoneme_vector_map(data, phoneme_dir):
    onehot_vector = np.zeros((len(data), len(phoneme_dir)))
    
    for i in range(len(data)):
        for j in range(len(phoneme_dir)):
            if data[i] == phoneme_dir[j]:
                onehot_vector[i,j] = 1
    
    return onehot_vector
    

"""
Main function
"""
if __name__ == "__main__":    
    
    # Windowing parameters
    window_size = 400
    window_shift = 160
    
    samp_rate = 16000
    bins = 40
    
    """
    Extracting audio data and converting to mel
    """
    #Speech files
    speech_file = ['AJJacobs_2007P-0001605-0003029']
    speech_file_list = ['AJJacobs_2007P-0001605-0003029', 'AJJacobs_2007P-0003133-0004110', 'AJJacobs_2007P-0004153-0005453']
    
    test_file = 'AJJacobs_2007P-0005613-0006448' # Take half of this file for testing
    
    # Get phoneme list (common for all files)
    with open('phonelist.txt', 'r') as infile:
        phoneme_dir = infile.readlines()
        
    for i in range(len(phoneme_dir)):
        phoneme_dir[i] = phoneme_dir[i].split()[0]
    
    unique_phonemes = []
    
    # For each file in the training set:
    for file_idx in range(len(speech_file_list)):
        
        # Get MFCCs for current file
        file_dir = 'wav/' + speech_file_list[file_idx] + '.wav'
        speech_mfcc = get_MFCC(file_dir, 40, window_size, window_shift)       
        print('Obtained MFCCs for file %d.\n' %(file_idx+1))
        
        # Get phonemes for that file
        phonemes = get_phonemes_data(speech_file_list[file_idx])
        
        # To see unique phonemes
        unique_phonemes = unique_phonemes + phonemes # Adding new set of phonemes
        unique_phonemes = list(set(unique_phonemes))
    
        """
        Form training data
        """
        
        X_train = speech_mfcc    
        Y_train = phoneme_vector_map(phonemes, phoneme_dir)
        
        if TRAIN:
            if file_idx == 0:   # First iteration, create model
                model = KerasPredictor.create_model(X_train, Y_train)
                print('Model Created.')
            else:
                model = load_model('phonemepredictor.h5')
                
            print('Training the model for file %d...' %(file_idx+1))
            model = KerasPredictor.train(model, X_train, Y_train)
            # Save the model
            model.save('phonemepredictor.h5')
            
   
    
    if TEST:
        # Get MFCCs for test file -- X_Train
        file_dir = 'wav/' + test_file + '.wav'
        X_test = get_MFCC(file_dir, 40, window_size, window_shift)       
        print('Obtained MFCCs for file %d.\n' %(file_idx+1))
        
        # Get phonemes for test file -- Y_Train
        test_phonemes = get_phonemes_data(test_file)
           
        Y_test = phoneme_vector_map(test_phonemes, phoneme_dir)
        
        # Changing size:
        end_idx = int(len(X_test)/2)
        X_test = X_test[:end_idx,:]
        Y_test = Y_test[:end_idx,:]
        
        print('Testing the model...')
        idx_pred = KerasPredictor.predict_phonemes('phonemepredictor.h5', X_test)
        
        # Convert to phonemes
        phoneme_pred = []
        for i in range(len(idx_pred)):
            phoneme_pred.append(phoneme_dir[np.argmax(idx_pred[i])])       
    
        # Test the accuraccy
        num_pred = 0
        
        for i in range(len(phoneme_pred)):
            if phoneme_pred[i] == test_phonemes[i]:
                num_pred += 1
        
#        for i in range(idx_pred):
#            if (idx_pred[i] == Y_test[i]).all():
#                num_pred += 1
        
        print('Test Accuracy = %.3f' %(num_pred/len(phoneme_pred)))