
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

read_data()