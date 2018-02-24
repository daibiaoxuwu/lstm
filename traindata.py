
from __future__ import print_function

import numpy as np
#import tensorflow as tf
#from tensorflow.contrib import rnn
import random
import collections
import time
import os
import json
import re
import word2vec

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
#writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
training_path = '/home/d/workspace/java/eclipse/work/lib/'
wordmodel = word2vec.load('/home/d/combine100.bin')
tagmodel=word2vec.load('/home/d/tagmodel100.bin')

def list_tags():
    mlist=os.listdir(training_path)
    for inputfile in mlist:
        if inputfile[-5:]=='.json':
            with open(os.path.join(training_path,inputfile)) as f:
                content = f.read()
                content=json.loads(content)
                content=content['sentences'][0]['parse']
                outtag=np.array()
                outword=np.array()
                while content!='':
                    print(content)
                    tag=re.search('\((\S+) (\S*)\)',content)
                    if tag==None:
                        print('error:',content)
                    tag1=tag.group(1)
                    outtag=outtag.hstack((outtag,tagmodel[tag.group(1)]))
                    if tag.group(2)!='':
                        outword=outword.hstack((outwork,tagmodel[tag.group(2)]))
                    content.replace(tag.group(0),'',1)



def read_data():
    mlist=os.listdir(training_path)
    inputfile=random.sample(mlist,1)[0]
    print(len(mlist))
    while inputfile[-5:]!='.json':
        print(inputfile[-5:])
        inputfile=random.sample(mlist,1)[0]

    with open(os.path.join(training_path,inputfile)) as f:
        content = f.read()
    print('loaded data')
    content=json.loads(content)
    print('parsed data')
    print(content['sentences'][0]['parse'])
list_tags
