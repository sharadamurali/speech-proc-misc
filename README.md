# Miscellaneous Speech Processing Programs
A collection of speech processing programs in python.

## LMS N-Step Predictor
Uses an LMS Algorithm to predict the next value of a noisy speech signal from the past N values. Non-linearlizing the the output with a sigmoid function improves performance and results in lower prediction error.

## Denoising Autoencoder
In this program, a number of clean speech files are downloaded, processed and mixed with a (weighted) noise signal, resulting in noisy speech. The mel spectrum of the audio data is used to train a neural network. The ratio of the mel spectra of the clean and noisy speech signal is used as output for training. The denoising autoencoder is tested on randomly split noisy data, as well as sequential audio data.

## GMM - Expectation Maximization
The gmm_em.py file contains an implementation of the Expectation-Maximization algorithm to fit data to a Gaussian Mixture Model. The gmm_nn.py trains a model to classify data points to the different gaussians.

## Phoneme Predictor
An implementation of a one-frame phoneme predictor. The predictor takes the MFCCs of the audio as input, and predicts the phoneme (as a one-hot vector).
