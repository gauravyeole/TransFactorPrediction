import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.layers.python.layers.layers import batch_norm,convolution2d,max_pool2d,fully_connected,flatten

class TextCNN(object):
    """
    A CNN text classifier
    Uses colvolutional layer followed by max-pooling and softmax layer
    """
    def __init__(self, sequence_length, num_classes):

        #placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        x_image = tf.reshape(self.input_x, shape=[-1, 14, 4, 1])

        n_conv1 = 44
        L_conv1 = 5
        maxpool_len1 = 2
        conv1 = convolution2d(x_image, n_conv1, [L_conv1, 4], padding='VALID', normalizer_fn=None)
        conv1_pool = max_pool2d(conv1, [maxpool_len1, 1], [maxpool_len1, 1])
        conv1_pool_len = int((101 - L_conv1 + 1) / maxpool_len1)

        # n_conv2 = n_conv1
        # L_conv2 = 3
        # maxpool_len2 = int(conv1_pool_len - L_conv2 + 1)  # global maxpooling (max-pool across temporal domain)
        # conv2 = convolution2d(conv1_pool, n_conv2, [L_conv2, 1], padding='VALID', normalizer_fn=None)
        # conv2_pool = max_pool2d(conv2, [maxpool_len2, 1], [maxpool_len2, 1])

        # LINEAR FC LAYER
        y_conv = fully_connected(flatten(conv1_pool), 2, activation_fn=None)
        prediction = tf.nn.softmax(y_conv)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=self.input_y))
        # train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
