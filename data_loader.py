import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from args import Config, Path


def data_generator(path_labels, path_dataset):
    config = Config()
    dir_annotation = os.listdir(path_labels)

    first = True
    for f in dir_annotation:
        if first:
            labels = np.load(os.path.join(path_labels, f))
            first = False
        else:
            temp = np.load(os.path.join(path_labels, f))
            labels = np.append(labels, temp, axis=0)
    labels = torch.from_numpy(labels)

    dataset_EEG_FpzCz = np.load(os.path.join(path_dataset, 'TF_EEG_Fpz-Cz_mean_std.npy')).astype('float32')
    dataset_EEG_PzOz = np.load(os.path.join(path_dataset, 'TF_EEG_Pz-Oz_mean_std.npy')).astype('float32')
    dataset_EOG = np.load(os.path.join(path_dataset, 'TF_EOG_mean_std.npy')).astype('float32')

    dataset = np.stack((dataset_EEG_FpzCz, dataset_EEG_PzOz, dataset_EOG), axis=1)
    dataset = torch.from_numpy(dataset)

    print('dataset: ', dataset.shape)

    # hold out the validation set
    X_train_test, X_val, y_train_test, y_val = train_test_split(dataset, labels, test_size=1/(config.num_fold+1), random_state=0, stratify=labels)

    val_set = TensorDataset(X_val, y_val)
    val_loader = DataLoader(dataset=val_set, batch_size=config.batch_size, shuffle=False)

    print('val_set:', len(X_val))
    return X_train_test, y_train_test, val_loader