import os
import scipy.io
from scipy.fftpack import fft
from scipy import signal
from math import log

import numpy as np
import h5py

channels = ['EEG_Fpz-Cz', 'EEG_Pz-Oz', 'EOG']

for i in channels:
    print(i)