
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
    print(len(mlist))
    for inputfile in mlist[7001:14000]:
        if inputfile[-4:]=='.txt':
            print(inputfile)
            with open(os.path.join(training_path,inputfile)) as f:
                resp = requests.post(url, f.read(), params=params).text
                try:
                    content=json.loads(resp)
                    for sentence in content['sentences']:
                        parse=sentence['parse'].replace('\n',' ')
                        with open('/home/d/parseword2','a+') as f:
                            f.write(parse+'\n')
                except:
                    print('error')
list_tags()
print('end')
