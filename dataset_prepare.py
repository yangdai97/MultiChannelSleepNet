import os
import shutil

import mne
import numpy as np

from args import Path


def prepare_SleepEDF_20(path_PSG, path_hypnogram, save_path):
    """extract 30-s epoch from EDF files"""
    for file in os.listdir(path_PSG):
        if "Hypnogram" in file:
            original_path = os.path.join(path_PSG, file)
            target_path = os.path.join(path_hypnogram, file)
            shutil.move(original_path, target_path)

    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}

    event_id_with_no_N3N4 = {'Sleep stage W': 1,
                             'Sleep stage 1': 2,
                             'Sleep stage 2': 3,
                             'Sleep stage R': 5}

    dir_PSG = os.listdir(path_PSG)
    dir_annotation = os.listdir(path_hypnogram)

    for i, j in zip(dir_PSG, dir_annotation):
        print('current file: ', i, j)

        PSG_file = os.path.join(path_PSG, i)
        annotation_file = os.path.join(path_hypnogram, j)

        raw_train = mne.io.read_raw_edf(PSG_file, stim_channel='marker', misc=['rectal'])
        annotation_train = mne.read_annotations(annotation_file)
        raw_train.set_annotations(annotation_train, emit_warning=False)

        annotation_train.crop(annotation_train[1]['onset'] - 30 * 60, annotation_train[-2]['onset'] + 30 * 60)
        raw_train.set_annotations(annotation_train, emit_warning=False)
        events_train, sleep_stage_exist = mne.events_from_annotations(raw_train, event_id=annotation_desc_2_event_id,
                                                                      chunk_duration=30.)

        tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

        if len(sleep_stage_exist) <= 4:
            epochs_train = mne.Epochs(raw=raw_train, events=events_train, event_id=event_id_with_no_N3N4, tmin=0.,
                                      tmax=tmax, baseline=None, preload=True)
        else:
            epochs_train = mne.Epochs(raw=raw_train, events=events_train, event_id=event_id, tmin=0., tmax=tmax,
                                      baseline=None, preload=True)

        X_train_eeg_FpzCz = epochs_train.copy().pick_channels(['EEG Fpz-Cz']).get_data()
        X_train_eeg_PzOz = epochs_train.copy().pick_channels(['EEG Pz-Oz']).get_data()
        X_train_eog = epochs_train.copy().pick_channels(['EOG horizontal']).get_data()
        y_train = epochs_train.copy().pick_channels(['EEG Fpz-Cz']).events[:, 2]
        y_train = y_train - 1

        for channel in ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG', 'labels']:
            if not os.path.exists(os.path.join(save_path, channel)):
                os.makedirs(os.path.join(save_path, channel))

        np.save(save_path + '/EEG_Fpz-Cz/{}_EEG_Fpz-Cz.npy'.format(i[0:12]), X_train_eeg_FpzCz)
        np.save(save_path + '/EEG_Pz-Oz/{}_EEG_Pz-Oz.npy'.format(i[0:12]), X_train_eeg_PzOz)
        np.save(save_path + '/EOG/{}_EOG.npy'.format(i[0:12]), X_train_eog)
        np.save(save_path + '/labels/{}_label.npy'.format(i[0:12]), y_train)


if __name__ == '__main__':
    path = Path()
    prepare_SleepEDF_20(path_PSG=path.path_PSG, path_hypnogram=path.path_hypnogram, save_path=path.path_raw_data)


