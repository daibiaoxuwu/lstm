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



with open(training_path) as f:
    resp = f.readlines()

#len:2071700
print('init:1')
def lemma(verb):
    url = 'http://127.0.0.1:9000'
    params = {'properties' : r"{'annotators': 'lemma', 'outputFormat': 'json'}"}
    resp = requests.post(url, verb, params=params).text
    content=json.loads(resp)
    return content['sentences'][0]['tokens'][0]['lemma']

def list_tags(st,step):
    realength=0
    tagdict={')':0}
    inputs=[]
    pads=[]
    answer=[]
    for sentence in resp[st:]:#一个sentence是一句话
        total=0
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in verbtags:
                    total+=1
        if total!=1:
            continue

        if len(answer)==step:
            break
        outword=[]
        for tag in sentence.split():
            if tag[0]=='(':
                if tag=='(MD':
                    mdflag=1
                else:
                    mdflag=0
                    if tag[1:] in verbtags:
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
        print('out+')
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
    print(answers.shape)
    return inputs,pads,answers


print('init:2')


# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
training_steps=10

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [training_steps, maxlength, embedding_size])
y = tf.placeholder("float", [training_steps, len(verbtags)])
p = tf.placeholder("float", [training_steps])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, len(verbtags)]))
}
biases = {
    'out': tf.Variable(tf.random_normal([len(verbtags)]))
}

def RNN(session,inputs, weights, biases, pads,answers):

    # reshape to [1, n_input]
    #print(tf.shape(x).eval())
    #x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    #x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    print('1')
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, p, dtype=tf.float32)
    print(tf.shape(outputs).eval())
#input: [10 1000 100]
#output: [10 1000 512]
    outputs=tf.transpose(outputs,[1,0,2])

    # there are n_input outputs but
    # we only want the last output
    pred=tf.matmul(outputs[-1], weights['out']) + biases['out']


# Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print(type(inputs),type(answers))
    print('2')
    session.run([optimizer, accuracy, cost, pred], feed_dict={x: inputs, y: answers })#???
    print('2')
    

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)
    #for training_iters in range(20717):
    for training_iters in range(207):
        # Generate a minibatch. Add some randomness on selection process.
        '''
        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
        '''
        inputs,pads,answers=list_tags(training_iters*training_steps,training_steps)
        print(inputs.shape)
        _, acc, loss, onehot_pred = RNN(session,inputs,weights,biases,pads,answers)
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            '''
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
            '''
        step += 1
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
