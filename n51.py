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

os.environ["CUDA_VISIBLE_DEVICES"]="0"#环境变量：使用第一块gpu
embedding_size=100#词向量维度数量
patchlength=0#神经网络的输入是一句只有一个动词的句子（以及其语法树），把动词变为原型，语法树的tag变为了VB。
#并预测它的动词时态。如果它不为0，输入变为这句话以及他前面的patchlength句话。
#语法树结构：（VB love）会被变为三个标签：（VB的（100维）one-hot标签，love的词向量标签，反括号对应的全0标签。
#每个反括号对应一个单独的标签，而正括号没有。
#one-hot的意思就是，假如有
maxlength=200#输入序列（包括语法树信息）的
verbtags=['VB','VBZ','VBP','VBD','VBN','VBG']

initial_learning_rate = 0.001
training_iters = 2000
training_steps=1
display_step = 1
saving_step=200

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

import reader
data=reader.reader(patchlength=patchlength)

# Target log path
logs_path = 'log/rnn_words'
writer = tf.summary.FileWriter(logs_path)

import model
model=rnnmodel.rnnmodel(batch_size=training_steps)

saver=tf.train.Saver()
# Launch the graph
print('start session')
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
with tf.Session(config=config) as session:
#with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    step = 0
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    count=patchlength
    while step < training_iters:
        count,inputs,pads,answers=data.list_tags(count,training_steps)
        #count,inputs,pads,answers=list_tags(0,training_steps)
        _, acc, loss, onehot_pred= session.run([model.optimizer, model.accuracy, model.cost, model.pred], \
                                                feed_dict={model.x: inputs, model.y: answers, model.p:pads})
        loss_total += loss
        acc_total += acc
        step += 1
        if step % display_step == 0:
            print("Iter= " + str(step+1) + ", used: "+str(count)+ ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step)," Elapsed time: ", elapsed(time.time() - start_time))
            start_time=time.time()
            acc_total = 0
            loss_total = 0
        if step % saving_step ==0:
            print(saver.save(session,'/home/djl/ckpt2/n5103.ckpt',global_step=step))
#            print(session.run(weights))
#            print(session.run(biases))
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
