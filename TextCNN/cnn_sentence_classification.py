
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
def load_dataset(filename):
    df = pd.read_csv(filename)
    df["np_features"] = df.apply(lambda  row: encode_dna_string(row["sequence"]), axis=1)
    df["np_label"] = df.apply(lambda  row: encode_label(row["label"]), axis=1)
    df = df.reindex(np.random.permutation(df.index))
    features = df.as_matrix(columns=[df["np_features"]])
    labels = df.as_matrix(columns=df["np_label"])
    return features, labels


def main(_):


    dropout_on = tf.placeholder(tf.float32)
    if dropout_on is not None:
        conv_keep_prob = 1.0
    else:
        conv_keep_prob = 1.0

    x = tf.placeholder(tf.float32, shape=[None, 14*4])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])

    x_image = tf.reshape([-1, 14, 4, 1])

    n_conv1 = 384 # TBD
    L_conv1 = 9 # TBD
    maxpool_len1 = 2
    conv1 = convolution2d(x_image, n_conv1, [L_conv1, 4], padding="VALID", normalizer_fn=None)
    conv1_pool_len = int((14-L_conv1+1)/maxpool_len1)

    n_conv2 = n_conv1
    L_conv2 = 5
    maxpool_len2 = int(conv1_pool_len - L_conv2 + 1)  # global maxpooling (max-pool across temporal domain)
    conv2 = convolution2d(conv1_pool, n_conv2, [L_conv2, 1], padding='VALID', normalizer_fn=None)
    conv2_pool = max_pool2d(conv2, [maxpool_len2, 1], [maxpool_len2, 1])
    # conv2_drop = tf.nn.dropout(conv2_pool, conv_keep_prob)

    # LINEAR FC LAYER
    y_conv = fully_connected(flatten(conv2_pool), 2, activation_fn=None)
    y_conv_softmax = tf.nn.softmax(y_conv)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())



if __name__ == '__main__':
    tf.app.run()