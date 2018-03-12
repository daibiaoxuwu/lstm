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
 
training_path='train/parseword' 
def list_tags(resp): 
    with open('train/resp','a+') as g: 
        st=0 
        en=1 
        while en<len(resp): 
            if(resp[en].strip()=='(ROOT'): 
                g.write(' '.join(resp[st:en]).replace('\n',' ')) 
                st=en 
            en+=1 
 
with open(training_path+'44') as f: 
    resp = f.readlines() 
    list_tags(resp) 
    ''' 
with open(training_path+'22') as f: 
    resp = f.readlines() 
    list_tags(resp,mm) 
with open(training_path+'33') as f: 
    resp = f.readlines() 
    list_tags(resp,mm) 
with open(training_path+'44') as f: 
    resp = f.readlines() 
    list_tags(resp,mm) 
    ''' 
