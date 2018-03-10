#encoding:utf-8
#n51test.py
#learning rate decay
#patchlength 0 readfrom resp
#add:saving session
from __future__ import print_function

print('init:0')
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
print('init:1')
#import random
#import collections
import time
import word2vec
print('init:2')
import os
#import json
import re
#import requests
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="0"#环境变量：使用第一块gpu
embedding_size=100#词向量维度数量
patchlength=0#神经网络的输入是一句只有一个动词的句子（以及其语法树），把动词变为原型，语法树的tag变为了VB。
#并预测它的动词时态。如果它不为0，输入变为这句话以及他前面的patchlength句话。
#语法树结构：（VB love）会被变为三个标签：（VB的（100维）one-hot标签，love的词向量标签，反括号对应的全0标签。
#每个反括号对应一个单独的标签，而正括号没有。
#one-hot的意思就是，假如有
maxlength=200#输入序列（包括语法树信息）的
verbtags=['VB','VBZ','VBP','VBD','VBN','VBG']

global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step, decay_steps=500,decay_rate=0.8)
training_iters = 11
training_steps=1
display_step = 10

# number of units in RNN cell
n_hidden = 512


start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = 'log/n51ot'
writer = tf.summary.FileWriter(logs_path)

print('loading model')
model=word2vec.load('train/combine100.bin')
print('loaded model')
# Text file containing words for training
training_path = r'train/resp'

max_acc=0



print('reading text')
with open(training_path) as f:
    resp=f.readlines()[:100]
print('read data:',len(resp))
print('loading lemma')

#len:2071700
'''
def lemma(verb):
    url = 'http://127.0.0.1:9000'
    params = {'properties' : r"{'annotators': 'lemma', 'outputFormat': 'json'}"}
    resp = requests.post(url, verb, params=params).text
    content=json.loads(resp)
    return content['sentences'][0]['tokens'][0]['lemma']
'''
with open('train/lemma2', 'rb') as f:
    ldict = pickle.load(f)
print('loaded lemma')

def lemma(verb):
    if verb in ldict:
        return ldict[verb]
    else:
        print('errverb:',verb)
        return verb

def list_tags(st,step):
    realength=0
    tagdict={')':0}
    inputs=[]
    pads=[]
    answer=[]
    count=st
    #fft=0
    for sentence in resp[st:]:#一个sentence是一句话
        if len(answer)==step:
            break
        outword=[]
        count+=1
        total=0
        
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in verbtags:
                    total+=1
        if total!=1:
            continue
        #else:
            #fft+=1
        
        for oldsentence in resp[count-patchlength:count]:
            for tag in oldsentence.split():
                if tag[0]=='(':
                    if tag not in tagdict:
                        tagdict[tag]=len(tagdict)
                    tagword=[0]*embedding_size
                    tagword[tagdict[tag]]=1
                    outword.append(tagword)
                else:                
                    node=re.match('([^\)]+)(\)*)',tag.strip())
                    if node:
                        if node.group(1) in model:
                            outword.append(model[node.group(1)].tolist())
                        else:
                            outword.append([0]*embedding_size)
                        tagword=[0]*embedding_size
                        tagword[0]=1
                        for _ in range(len(node.group(2))-1):
                            outword.append(tagword)

        
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
        outword=np.array(outword)
        if outword.shape[0]>maxlength:
            print('pass')
            answer=answer[:-1]
            continue
        pads.append(outword.shape[0])
        outword=np.pad(outword,((0,maxlength-outword.shape[0]),(0,0)),'constant')
        inputs.append(outword)
    inputs=np.array(inputs)
    answers=np.zeros((len(answer),len(verbtags)))
    for num in range(len(answer)):
        answers[num][answer[num]]=1
    #print(fft)
    return count,inputs,pads,answers

#dictionary, reverse_dictionary = build_dataset(training_data)
#vocab_size = len(dictionary)

# Parameters
vocab_size=len(verbtags)
# tf Graph input
x = tf.placeholder("float", [training_steps, maxlength, embedding_size])
y = tf.placeholder("float", [training_steps, len(verbtags)])
p = tf.placeholder("float", [training_steps])

# RNN output node weights and biases
weights = tf.Variable(tf.random_normal([256, vocab_size])) 
biases =  tf.Variable(tf.random_normal([vocab_size])) 

def RNN(x, p, weights, biases):
    #x = tf.reshape(x, [-1, maxlength])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    #x = tf.split(x,maxlength,1)
    #x=tf.unstack(x,maxlength,axis=1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(512),rnn.BasicLSTMCell(256)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    initial_state = rnn_cell.zero_state(training_steps, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, sequence_length=p, initial_state=initial_state, dtype=tf.float32)
    outputs = tf.stack(outputs)
 #   outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    #batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    #index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    index = tf.range(0, tf.shape(outputs)[0],dtype=tf.int32) * maxlength
    print(index.shape)
    index=tf.add(index , tf.cast((p - 1),tf.int32))
    print(index.shape)
    # Indexing
    print(outputs.shape)
    outputs = tf.gather(tf.reshape(outputs, [-1, 256]), index)
    print(outputs.shape)
#    pr=tf.Print(outputs[0][-1],[outputs[0][-1]])

#input: [10 1000 100]
#output: [10 1000 512]
#    outputs=tf.transpose(outputs,[1,0,2])

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs, weights) + biases

pred = RNN(x, p, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver=tf.train.Saver()
print('ready')

# Launch the graph
print('start session')
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
with tf.Session(config=config) as session:
#with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state('/home/djl/ckpt3/')
    print(ckpt)
    saver.restore(session, ckpt.model_checkpoint_path)  
    print('session init')
    print(session.run(weights))
    print(session.run(biases))
    step = 0
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    count=patchlength
    while step < training_iters:
        count,inputs,pads,answers=list_tags(count,training_steps)
        #count,inputs,pads,answers=list_tags(0,training_steps)
        if count>=len(resp):
            count=patchlength
            print('epoch')
            continue
        _, acc, loss, onehot_pred= session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: inputs, y: answers, p:pads})
        loss_total += loss
        acc_total += acc
        '''
        if (step+1) % 100==0:
            learning_rate*=0.9
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
        '''
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", used: "+str(count)+ ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step)," Elapsed time: ", elapsed(time.time() - start_time))
            start_time=time.time()
            acc_total = 0
            loss_total = 0
        if(step+1) % 100 ==0:
            print(session.run(weights))
            print(session.run(biases))
        step += 1
        global_step += 1
    #    print(global_step.eval())
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
