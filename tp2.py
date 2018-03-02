#encoding:utf-8
#n4.py:dynamic rnn
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

training_path = '../parseword'
maxlength=1000
verbtags=['VB','VBZ','VBP','VBD','VBN','VBG']

def list_tags(resp,mm):
    for sentence in resp:#一个sentence是一句话
        total=0
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in verbtags:
                    total+=1
        if total!=1:
            continue
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in verbtags:
                    mm[verbtags.index(tag[1:])]+=1
    print(mm)
    return
def output(resp,mi,mm):
    for sentence in resp:
        total=0
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in verbtags:
                    total+=1
        if total!=1:
            continue
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in verbtags:
                    mm[verbtags.index(tag[1:])]+=1
                    if mm[verbtags.index(tag[1:])]<=mi:
                        with open('../resp','a+') as g:
                            g.write(sentence)
    print(mm)
    return


model=word2vec.load('../combine100.bin')
mm=[0]*6
with open(training_path+'11') as f:
    resp = f.readlines()
    list_tags(resp,mm)
with open(training_path+'22') as f:
    resp = f.readlines()
    list_tags(resp,mm)
with open(training_path+'33') as f:
    resp = f.readlines()
    list_tags(resp,mm)
with open(training_path+'44') as f:
    resp = f.readlines()
    list_tags(resp,mm)
mi=min(mm)
mm=[0]*6
with open(training_path+'11') as f:
    resp = f.readlines()
    output(resp,mi,mm)
with open(training_path+'22') as f:
    resp = f.readlines()
    output(resp,mi,mm)
with open(training_path+'33') as f:
    resp = f.readlines()
    output(resp,mi,mm)
with open(training_path+'44') as f:
    resp = f.readlines()
    output(resp,mi,mm)

