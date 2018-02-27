
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
training_path = '/home/d/Downloads/p2r'


def list_tags():
    url = 'http://127.0.0.1:9000'
    params = {'properties' : r"{'annotators': 'tokenize,ssplit,pos,lemma,parse', 'outputFormat': 'json'}"}
    tagdict={')':0}
    inputs=[]
    mlist=os.listdir(training_path)
    print('1')
    for inputfile in mlist[:10]:
        if inputfile[-4:]=='.txt':
            print(inputfile)
            with open(os.path.join(training_path,inputfile)) as f:
                resp = requests.post(url, f.read(), params=params).text
                content=json.loads(resp)
                for sentence in content['sentences']:
                    parse=sentence['parse']
                    outword=[]
                    for tag in parse.split():
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
                                for _ in node.group(2):
                                    outword.append(tagword)
                print(outword)
list_tags()
