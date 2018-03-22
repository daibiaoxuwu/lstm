#encoding:utf-8 
#n4.py:dynamic rnn 
from __future__ import print_function 
 
import numpy as np 
import re 
import time
 
training_path='train/parseword' 
def list_tags(resp): 
    print(len(resp))
    print(re.sub('\s',' ',resp[0]))
    with open('train/resp2','a+') as g: 
        st=0
        en=1
#        st=len(resp)-100 
#        en=len(resp)-99
        while en<len(resp): 
            if(resp[en].strip()[:5]=='(ROOT'): 
                g.write(re.sub('\s+',' ',(' '.join(resp[st:en]).replace('\n',' ')))+'\n')
                st=en 
            en+=1 
 
with open(training_path+'11') as f: 
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
