#t1.py
#从一个大的纯parse文件parseword拿输入
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
# listtags函数制造的输入数据的"每一个句字"的最大长度
maxlength=5000



def lemma(verb):
    url = 'http://127.0.0.1:9000'
    params = {'properties' : r"{'annotators': 'lemma', 'outputFormat': 'json'}"}
    resp = requests.post(url, verb, params=params).text
    content=json.loads(resp)
    return content['sentences'][0]['tokens'][0]['lemma']

def list_tags():
    tagdict={')':0}
    inputs=[]
    pads=[]
    answer=[]
    verbtags=['VB','VBZ','VBP','VBD','VBN','VBG']
    with open(training_path) as f:
        resp = f.readlines()
        for sentence in resp:#一个sentence是一句话
            singleverb=0
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
                                    node=lemma(node.group(1))
                                    outword.append(model[node].tolist())
                                else:
                                    outword.append(model[node.group(1)].tolist())
                            else:
                                outword.append([0]*embedding_size)
                            tagword=[0]*embedding_size
                            tagword[0]=1
                            for _ in range(node.group(2))-1:
                                outword.append(tagword)
                outword=np.array(outword)
                pads.append(outword.shape[0])
                outword=np.pad(outword,((0,maxlength-outword.shape[0]),(0,0)),'constant')
                inputs=np.stack((inputs,outword))

list_tags()
print('end')
