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
training_path = '/home/d/parseword'
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
    top=[0]*40
    for sentence in resp:#一个sentence是一句话
        total=0
        outword=[]
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in verbtags:
                    total+=1
        try:
            top[total]+=1
        except:
            print(top)
    print(top)
list_tags()

