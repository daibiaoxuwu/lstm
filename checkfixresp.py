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
    with open('train/resp') as g: 
        big=g.readlines()
        print('resplength',len(big))
        st=0 
        en=1 
        while en<len(resp): 
            if(resp[en].strip()=='(ROOT'): 
                gg=' '.join(resp[st:en]).replace('\n',' ') 
                print(big.index(gg))
                break
            en+=1
 
with open(training_path+'22') as f: 
    resp = f.read().split()
    print('parsword',len(resp))
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
