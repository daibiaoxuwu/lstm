#encoding:utf-8
#input.py 
#learning rate decay
#patchlength 0 readfrom resp
#add:saving session
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
import pickle

class reader(object):
    def __init__(self,\
                patchlength=0,\
                maxlength=200,\
                embedding_size=100):
#patchlength=0#神经网络的输入是一句只有一个动词的句子（以及其语法树），把动词变为原型，语法树的tag变为了VB。
#并预测它的动词时态。如果它不为0，输入变为这句话以及他前面的patchlength句话。
#语法树结构：（VB love）会被变为三个标签：（VB的（100维）one-hot标签，love的词向量标签，反括号对应的全0标签。
#每个反括号对应一个单独的标签，而正括号没有。
#one-hot的意思就是，假如有
#maxlength=200#输入序列（包括语法树信息）的
        self.patchlength=patchlength
        self.maxlength=maxlength
        self.embedding_size=embedding_size
        self.verbtags=['VB','VBZ','VBP','VBD','VBN','VBG']
        self.model=word2vec.load('train/combine100.bin')
        print('loaded model')

        with open(r'train/resp') as f:
            self.resp=f.readlines()[:100]
        print('loaded data:',len(self.resp))

        with open('train/lemma2', 'rb') as f:
            self.ldict = pickle.load(f)
        print('loaded lemma')

    def lemma(self,verb):
        if verb in self.ldict:
            return self.ldict[verb]
        else:
            print('errverb:',verb)
            return verb

    def list_tags(self,st,step):
        realength=0
        tagdict={')':0}
        inputs=[]
        pads=[]
        answer=[]
        count=st
        while True:#防止读到末尾
            for sentence in self.resp[st:]:#一个sentence是一句话
                if len(answer)==step:
                    break
                outword=[]
                count+=1
                total=0
                
                for tag in sentence.split():
                    if tag[0]=='(':
                        if tag[1:] in self.verbtags:
                            total+=1
                if total!=1:
                    continue
                
                for oldsentence in self.resp[count - self.patchlength:count]:
                    for tag in oldsentence.split():
                        if tag[0]=='(':
                            if tag not in tagdict:
                                tagdict[tag]=len(tagdict)
                            tagword=[0]*self.embedding_size
                            tagword[tagdict[tag]]=1
                            outword.append(tagword)
                        else:                
                            node=re.match('([^\)]+)(\)*)',tag.strip())
                            if node:
                                if node.group(1) in self.model:
                                    outword.append(self.model[node.group(1)].tolist())
                                else:
                                    outword.append([0]*self.embedding_size)
                                tagword=[0]*self.embedding_size
                                tagword[0]=1
                                for _ in range(len(node.group(2))-1):
                                    outword.append(tagword)

                
                for tag in sentence.split():
                    if tag[0]=='(':
                        if tag=='(MD':
                            mdflag=1
                        else:
                            mdflag=0
                            if tag[1:] in self.verbtags:
                                answer.append(self.verbtags.index(tag[1:]))
                                tag='(VB'
                                vbflag=1
                            else:
                                vbflag=0
                            if tag not in tagdict:
                                tagdict[tag]=len(tagdict)
                            tagword=[0]*self.embedding_size
                            tagword[tagdict[tag]]=1
                            outword.append(tagword)
                    else:
                        if mdflag==0:
                            node=re.match('([^\)]+)(\)*)',tag.strip())
                            if node:
                                if node.group(1) in self.model:
                                    if vbflag==1:
                                        node2=self.lemma(node.group(1))
                                        if node2 in self.model:
                                            outword.append(self.model[node2].tolist())
                                        else:
                                            outword.append([0]*self.embedding_size)
                                    else:
                                        outword.append(self.model[node.group(1)].tolist())
                                else:
                                    outword.append([0]*self.embedding_size)
                                tagword=[0]*self.embedding_size
                                tagword[0]=1
                                for _ in range(len(node.group(2))-1):
                                    outword.append(tagword)
                outword=np.array(outword)
                if outword.shape[0]>self.maxlength:
                    print('pass')
                    answer=answer[:-1]
                    continue
                pads.append(outword.shape[0])
                outword=np.pad(outword,((0,self.maxlength-outword.shape[0]),(0,0)),'constant')
                inputs.append(outword)
            inputs=np.array(inputs)
            answers=np.zeros((len(answer),len(self.verbtags)))
            for num in range(len(answer)):
                answers[num][answer[num]]=1
            if count>=len(self.resp):
                count=self.patchlength
                print('epoch')
#continue the 'while True' loop
            else:
                return count,inputs,pads,answers
