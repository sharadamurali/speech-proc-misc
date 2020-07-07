# -*- coding: utf-8 -*-
"""
ExtractAudio:
    get_audio: extracts and converts the audio file (.wav) from the provided URL
    mix_audio: mixes two audio files from input URLs
    get_RAW : returns the RAW contents of the .wav file
"""

import scipy.io.wavfile
import subprocess
import youtube_dl
import numpy as np

def get_audio(audio_url, file_name, start_trim):
    # Defining file names
    file_name_out = file_name + '.%(ext)s'
    final_file = file_name + '_trim'
    
    # SOX commands
    sox_command_resample  = ['sox', file_name + '.wav', '-c 1', '-b 16', '-r 16k', file_name + '_new.wav']
    sox_command_trim = ['sox', file_name + '_new.wav', final_file + '.wav', 'trim', start_trim, '00:05:00']
    
    # YoutubeDL options
    ydl_opts = {
    'format': 'bestaudio/best', 
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'quiet': True,
    'outtmpl': file_name_out,
    'restrictfilenames': True}
    
    # Downloading the speech audio from URL
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    print('Downloading file from URL...')
    ydl.download(audio_url)
    print('Done.')
    
    # Converting and trimming using SOX
    print('Converting to 16kHz 16bit mono channel .wav file')
    subprocess.call(sox_command_resample)
    subprocess.call(sox_command_trim)
    print('Done.')
    
    return final_file
    
def get_RAW(file):
    rate, raw_data = scipy.io.wavfile.read(file + '.wav')
#    raw_data = np.array(raw_data_0, dtype = 'f')
#    raw_data /= np.abs(np.max(raw_data))
    
    return raw_data

def mix_audio(file_1, file_2):
    file_out =  file_1 + '_' + file_2
    print('Mixing Audio...')
    sox_mix_command = ['sox', '-m', file_1 + '.wav', file_2 + '.wav', file_out + '.wav']
    subprocess.call(sox_mix_command)
    print('Done.')
    return file_out

