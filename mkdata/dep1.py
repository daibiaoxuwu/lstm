
from __future__ import print_function

import numpy as np
import time
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
embedding_size=100

# Text file containing words for training


def list_tags():
    url = 'http://166.111.139.15:9000'
    params = {'properties' : r"{'annotators': 'tokenize,ssplit,pos,lemma,parse,depparse', 'outputFormat': 'json'}"}
    tot=0
    fla=1
    tot2=0
    with open('../train/combine.txt') as f:
        a=f.readline()
        while a!='':
            resp = requests.post(url,a,params=params).text
            try:
                content=json.loads(resp)
                for sentence in content['sentences']:
                    for i in sentence['enhancedPlusPlusDependencies']:
                        if 'nsubjpass' in i.values():
                            tot+=fla
                            fla=0
                            if tot==1000:
                                print(tot2/1000)
                                input()
#                        parse=sentence['parse'].replace('\n',' ')
#                        with open('dep1','a+') as f:
#                            f.write(parse+'\n')
            except:
                raise NameError
                print('error')
            fla=1
            tot2+=1
            a=f.readline()
list_tags()
print('end')
