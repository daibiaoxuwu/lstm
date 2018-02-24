#train.py
#直接修改输入文件的语法树.变成冗余词汇.

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
write_path = '/home/d/workspace/java/eclipse/work/write/'
wordmodel = word2vec.load('/home/d/combine100.bin')
tagmodel=word2vec.load('/home/d/tagmodel100.bin')
#tagmodel=word2vec.load('/home/d/1.bin')

def list_tags():
    set1=set()
    mlist=os.listdir(training_path)
    output=''
    count=0
#所有的话放在一起
    for inputfile in mlist:
        count+=1
        if count%100==0:
            print(count)
        if inputfile[-5:]=='.json':
            with open(os.path.join(training_path,inputfile)) as f:
                full = f.read()
                full=json.loads(full)
                for sentence in full['sentences']:
                    content=sentence['parse'].split()
                    stack=[]
                    for tag in content:
                        if tag[0]=='(':
                            tag=tag[1:]
                            stack.append(tag)
                            try:
                                aa=tagmodel[''.join(stack)]
                            except:
                                print(''.join(stack))
                        else:
                            stack=stack[:-1]
                    output+='\n'

                
                

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

list_tags()
