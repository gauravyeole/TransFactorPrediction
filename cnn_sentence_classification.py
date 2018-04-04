
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.layers.python.layers.layers import batch_norm,convolution2d,max_pool2d,fully_connected,flatten
import tensorflow as tf
import numpy as np
import pandas as pd



sess = tf.InteractiveSession()

def encode_label(label):
    label = np.empty(2, dtype=np.dtype('float32'))
    if bound == 0:
        label[0] = 0.0
        label[1] = 0.0
    else:
        label[0] = 0.0
        label[1] = 1.0
    return label

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
    return features

# argument: input filename
def encode_dataset(filename):
    df = pd.read_csv(filename)
    df["np_features"] = df.apply(lambda  row: encode_dna_string(row["sequence"]), axis=1)
    df["np_label"] = df.apply(lambda  row: encode_label(row["label"]), axis=1)
    features = df.as_matrix(columns=[df["np_features"]])
    labels = df.as_matrix(columns=df["np_label"])
    return (features, labels)


def main(_):
    global _train_epochs_completed
    global _validation_epochs_completed
    global _test_epochs_completed
    global _datasets
    global _validation_size
    global _test_labels

    dropout_on = tf.placeholder(tf.float32)
    if dropout_on is not None:
        conv_keep_prob = 1.0
    else:
        conv_keep_prob = 1.0



if __name__ == '__main__':
    tf.app.run()