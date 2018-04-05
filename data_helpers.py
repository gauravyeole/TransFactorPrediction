from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.layers.python.layers.layers import batch_norm,convolution2d,max_pool2d,fully_connected,flatten
import tensorflow as tf
import numpy as np
import pandas as pd

def encode_label(bound):
    label = np.empty(2, dtype=np.dtype('float32'))
    if bound == 0:
        label[0] = 0.0
        label[1] = 0.0
    else:
        label[0] = 0.0
        label[1] = 1.0
    return label.tolist()

def encode_dna_string(dna_string):
    num_bases = len(dna_string)*4
    features = np.empty(num_bases, dtype=np.dtype('float32'))
    cur_base = 0
    for dna_base in dna_string:
        if dna_base is 'A':
            features[cur_base + 0] = 1.0
            features[cur_base + 1] = 0.0
            features[cur_base + 2] = 0.0
            features[cur_base + 3] = 0.0
        if dna_base is 'T':
            features[cur_base + 0] = 0.0
            features[cur_base + 1] = 1.0
            features[cur_base + 2] = 0.0
            features[cur_base + 3] = 0.0
        if dna_base is 'G':
            features[cur_base + 0] = 0.0
            features[cur_base + 1] = 0.0
            features[cur_base + 2] = 1.0
            features[cur_base + 3] = 0.0
        if dna_base is 'C':
            features[cur_base + 0] = 0.0
            features[cur_base + 1] = 0.0
            features[cur_base + 2] = 0.0
            features[cur_base + 3] = 1.0
        cur_base = cur_base + 4
    return features.tolist()

def load_data_and_labels(filename):
    fullpath = "/Users/gauravyeole/Downloads/BigData/TransFactorPrediction/" + filename
    df = pd.read_csv(fullpath, usecols=['id', 'sequence', 'label'])
    df['np_features'] = df.sequence.apply(encode_dna_string)
    df['np_label'] = df.label.apply(encode_label)
    df = df.reindex(np.random.permutation(df.index))
    features = df.np_features.as_matrix().tolist()
    labels = df.np_label.as_matrix().tolist()
    return np.array(features), np.array(labels)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]