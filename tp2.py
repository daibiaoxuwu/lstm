#encoding:utf-8
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

print('init:0')
start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = 'log/rnn_words'
writer = tf.summary.FileWriter(logs_path)
embedding_size=100

model=word2vec.load('/home/d/combine100.bin')
# Text file containing words for training
training_path = '/home/d/parseword'
maxlength=1000
verbtags=['VB','VBZ','VBP','VBD','VBN','VBG']



print('init:1')
def lemma(verb):
    url = 'http://127.0.0.1:9000'
    params = {'properties' : r"{'annotators': 'lemma', 'outputFormat': 'json'}"}
    resp = requests.post(url, verb, params=params).text
    content=json.loads(resp)
    return content['sentences'][0]['tokens'][0]['lemma']

def list_tags():
    realength=0
    tagdict={')':0}
    inputs=[]
    pads=[]
    answer=[]
    with open(training_path) as f:
        resp = f.readlines()
        for sentence in resp[:100]:#一个sentence是一句话
            singleverb=0
            outword=[]
            for tag in sentence.split():
                if tag[0]=='(':
                    if tag=='(MD':
                        mdflag=1
                    else:
                        mdflag=0
                        if tag[1:] in verbtags:
                            if singleverb==1:
                                break
                            singleverb=1
                            answer.append(verbtags.index(tag[1:]))
                            tag='(VB'
                            vbflag=1
                        else:
                            vbflag=0
                        if tag not in tagdict:
                            tagdict[tag]=len(tagdict)
                        tagword=[0]*embedding_size
                        tagword[tagdict[tag]]=1
                        outword.append(tagword)
                else:
                    if mdflag==0:
                        node=re.match('([^\)]+)(\)*)',tag.strip())
                        if node:
                            if node.group(1) in model:
                                if vbflag==1:
                                    node2=lemma(node.group(1))
                                    if node2 in model:
                                        outword.append(model[node2].tolist())
                                    else:
                                        outword.append([0]*embedding_size)
                                else:
                                    outword.append(model[node.group(1)].tolist())
                            else:
                                outword.append([0]*embedding_size)
                            tagword=[0]*embedding_size
                            tagword[0]=1
                            for _ in range(len(node.group(2))-1):
                                outword.append(tagword)
            outword=np.array(outword)
            if realength<outword.shape[0]:
                realength=max(realength,outword.shape[0])
                print(realength)
            if realength>maxlength:
                print('lengtherror')
            pads.append(outword.shape[0])
            outword=np.pad(outword,((0,maxlength-outword.shape[0]),(0,0)),'constant')
            inputs.append(outword)
    inputs=np.array(inputs)
    answers=np.zeros((len(answer),len(verbtags)))
    for num in range(len(answer)):
        answers[num][answer[num]]=1
    return inputs,pads,answers


inputs,pads,answers=list_tags()
vocab_size = embedding_size
print('init:2')


# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, len(verbtags)]))
}
biases = {
    'out': tf.Variable(tf.random_normal([len(verbtags)]))
}

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    print(tf.shape(x))
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)
    print(inputs.shape)

