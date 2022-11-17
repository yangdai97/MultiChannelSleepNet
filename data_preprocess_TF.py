import os
import numpy as np

from scipy.fftpack import fft
from scipy import signal
from tqdm import tqdm

from args import Path


def data_array_concat(path_array):
    """concat data from each subject"""
    dir_PSG = os.listdir(path_array)
    first = True
    print('Preparing dataset:')
    for f in tqdm(dir_PSG):
        if first:
            data_channel = np.load(os.path.join(path_array, f)).astype('float32')
            first = False
        else:
            temp = np.load(os.path.join(path_array, f)).astype('float32')
            data_channel = np.append(data_channel, temp, axis=0)
    data_channel = np.squeeze(data_channel, axis=1)
    return data_channel


def spectrogram(x, window, n_overlap, nfft):
    """
    Transform to time-frequency images. This function imitates function spectrogram in Matlab
    Args:
        x (numpy array): Data
        window (int): Size of window function
        n_overlap (int):Number of coincidence points between two segments
        nfft (int): Number of points during Fast Fourier Transform
    """
    len_x = len(x)
    step = window - n_overlap
    nn = nfft // 2 + 1
    num_win = int(np.floor((len_x - n_overlap) / (window - n_overlap)))
    spectrogram_data = []
    # Hamming window default
    win = signal.hamming(window)
    for i in range(num_win):
        subdata = x[i * step: i * step + window]
        F = fft(subdata * win, n=nfft)
        spectrogram_data.append(F[:nn])
    spectrogram_data = np.array(spectrogram_data)
    return spectrogram_data


def data_normalize(dataset, channel):
    """normalize datasets of each channel to zero mean and unit variance"""
    for i in tqdm(range(dataset.shape[0])):
        if True in np.isinf(dataset[i]):
            for j in range(29):
                if True in np.isinf(dataset[i][j]):
                    for k in range(128):
                        if np.isinf(dataset[i][j][k]):
                            if k != 127:
                                print('location of inf: ', i, ',', j, ',', k)
                            if k == 0:
                                if j == 0:
                                    dataset[i][j][k] = dataset[i][j+1][k]
                                else:
                                    dataset[i][j][k] = dataset[i][j-1][k]
                            else:
                                dataset[i][j][k] = dataset[i][j][k-1]

    dataset = (dataset - np.mean(dataset)) / np.std(dataset)

    ans1 = np.isinf(dataset)
    ans2 = np.isnan(dataset)

    if not ((True in ans1) and (True in ans2)):
        np.save('./data/sleepEDF-78/data_array/TF_data/TF_{}_mean_std.npy'.format(channel), dataset)


if __name__ == '__main__':
    path = Path()

    fs = 100
    overlap = 1
    nfft = 256
    win_size = 2

    for channel in ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']:
        print('-' * 15, 'Processing channel:{}'.format(channel), '-' * 15)
        data_channel = data_array_concat(path_array=os.path.join(path.path_raw_data, channel))
        X = np.zeros([data_channel.shape[0], 29, int(nfft / 2)])
        print('Transform to TF images:')
        for i in tqdm(range(data_channel.shape[0])):
            Xi = spectrogram(data_channel[i, :], win_size * fs, overlap * fs, nfft)
            Xi = 20 * np.log10(abs(Xi))
            X[i, :, :] = Xi[:, 1:129]

        print('Normalize:')
        data_normalize(dataset=X, channel=channel)
