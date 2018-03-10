#encoding:utf-8
#n51.py 
#learning rate decay
#patchlength 0 readfrom resp
#add:saving session
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
import word2vec
import os
import json
import re
import requests
import pickle


class rnnmodel(object):
    def __init__(self,\
                vocab_size=6,\
                maxlength=200,\
                embedding_size=100,\
                initial_training_rate=0.001,\
                batch_size=1):
# Target log path
        self.logs_path = 'log/rnn_words'
#learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(initial_training_rate, global_step=self.global_step, decay_steps=500,decay_rate=0.8)
# tf Graph input
        self.x = tf.placeholder("float", [batch_size, maxlength, embedding_size])
        self.y = tf.placeholder("float", [batch_size, vocab_size])
        self.p = tf.placeholder("float", [batch_size])
# RNN output node weights and biases
        self.weights = tf.Variable(tf.random_normal([256, vocab_size])) 
        self.biases =  tf.Variable(tf.random_normal([vocab_size])) 

        def RNN(x, p, weights, biases):
            rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(512),rnn.BasicLSTMCell(256)])
            initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, sequence_length=p, initial_state=initial_state, dtype=tf.float32)
            outputs = tf.stack(outputs)
            index = tf.range(0, tf.shape(outputs)[0],dtype=tf.int32) * maxlength
            index=tf.add(index , tf.cast((p - 1),tf.int32))
            outputs = tf.gather(tf.reshape(outputs, [-1, 256]), index)
            return tf.matmul(outputs, weights) + biases

# generate prediction
        self.pred = RNN(self.x, self.p, self.weights, self.biases)

# Loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

# Model evaluation
        self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

