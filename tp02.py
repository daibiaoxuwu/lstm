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
patchlength=3

# Text file containing words for training
training_path = 'D:\djl\parseword11'
maxlength=700
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

def list_tags():
    with open('lemma','w') as g:
        for sentence in resp:#一个sentence是一句话
            outword=''
            for tag in sentence.split():
                if tag[0]!='(':
                    node=re.match('([^\)]+)(\)*)',tag.strip())
                    if node:
                        outword+=lemma(node.group(1))
                        outword+='\n'
                    else:
                        print('error:',tag.strip())
            g.write(outword)
    return
list_tags()
