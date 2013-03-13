import matplotlib as mpl
import matplotlib.pyplot as pp
import matplotlib.pylab as pl
from matplotlib.mlab import specgram

import numpy as np

import csv
import aifc

from local_settings import WHALE_HOME
from progressbar import RealProgressBar, withProgress

data_files = []
labels = []
all_cases = []
test_cases = np.arange(54503)

with open("%s/data/train.csv"%WHALE_HOME, 'r') as csvfile:
    label_reader = csv.reader(csvfile)
    label_reader.next() # drop the headings
    for index, (file_name, label) in enumerate(label_reader):
        index = int(index)
        data_files.append(file_name)
        labels.append(1 if label=='1' else 0)
    all_cases = np.array(range(0, len(data_files)))
    labels = np.array(labels)

whale_cases = all_cases[labels==1]
no_whale_cases = all_cases[labels==0]

# Some spectrogram parameters

short_specgram = { 'detrend':mpl.mlab.detrend_mean,
                   'NFFT': 128, 
                   'Fs':2, 
                   'noverlap': 0}

long_specgram = { 'detrend':mpl.mlab.detrend_mean, 
                  'NFFT':256, 
                  'Fs':2, 
                  'noverlap':178 }

pca_specgram = short_specgram

def load_aiff(filename):
    snd = aifc.open(filename)
    snd_string = snd.readframes(snd.getnframes())
    snd.close()
    # need to do .byteswap as aifc loads / converts to a bytestring in
    # MSB ordering, but numpy assumes LSB ordering.
    return np.fromstring(snd_string, dtype=np.uint16).byteswap()

def load_test_case(n, normalised=True):
    if n<0 or n>=54503:
        raise ValueError("test case out of range: %d" % n )
    else:
        filename = "%s/data/test/test%d.aiff" % (WHALE_HOME, n+1)
        s = load_aiff(filename)
        if normalised:
            s = s/float(np.max(np.abs(s)))
        return s
        
def load_training_case(n, normalised=True):
    if n < 0 or n >= 30000:
        raise ValueError("training case out of range: %d" % n)
    else:
        filename = "%s/data/train/%s" % (WHALE_HOME, data_files[n])
        s = load_aiff(filename)
        if normalised:
            s = s/float(np.max(np.abs(s)))
        return s

def load_case(n, training):
    if training:
        data = load_training_case(n)
    else:
        data = load_test_case(n)
    return data

def get_spectrogram(n, training=True, spec_opts=long_specgram):
    data = load_case(n, training)
    s,f,t = specgram(data, **spec_opts)
    return s

def get_log_spectrogram(n, training=True, spec_opts=long_specgram):
    data = load_case(n, training)
    s,f,t = specgram(data, **spec_opts)
    return np.log(s)

def visualize_cases(cases=all_cases, load_function=get_spectrogram):
    for n in cases:
        pl.clf()
        d_long=load_function(n)
        pl.subplot(211)
        pl.imshow(d_long, aspect='auto')
        pl.subplot(212)
        pl.hist(d_long.flatten(), bins=100)
        pl.xlim([-12, 2])
        if raw_input('q to stop; anything else to continue...') == 'q':
            break

def calculate_mean_over_class(cases=all_cases, load_function=get_spectrogram):
    N = len(cases)
    mean_data = load_function(cases[0])
    for ind, n in enumerate(withProgress(cases[1:], 
                                RealProgressBar())):
        s = load_function(n)
        mean_data = vemean_data + (s - mean_data)/(float (ind+2))

    return mean_data

def translate_and_project_onto_vector(cases, t_vec, p_vec, load_function=get_spectrogram):
    t_vec = t_vec.flatten()
    p_vec = p_vec.flatten()
    mag = np.sqrt(np.dot(p_vec, p_vec))

    return [ np.dot(load_function(n).flatten() \
                        - t_vec, p_vec)/mag 
             for n in withProgress(cases, RealProgressBar())]
    

def load_data_subset(cases,
                     load_function=get_training_case):
    labels_subset = labels[cases]
    data_subset = np.array([load_function(n).flatten() 
                   for n in withProgress(cases,
                                         RealProgressBar())])
    return data_subset, labels_subset
