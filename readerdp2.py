#encoding:utf-8
#reader.py 只看含有一个动词的句子(十分之一左右)

import numpy as np
import word2vec
import re
import time
import os
import pickle
import random
import requests
import json
from queue import Queue
#bug:shorten和shorten_front不一样的话,每一遍都得重新计算而不是直接从队列里拿出来!


class reader(object):
    def printtag(self,number):
#        return [k for k, v in self.verbtags.items() if v == number][0]
        return self.verbtags[number]
    def parse(self,text):
        #print('start parse')
        if(text==''):
            raise NameError
        url = 'http://166.111.139.15:9000'
        params = {'properties' : r"{'annotators': 'tokenize,ssplit,pos,lemma,parse', 'outputFormat': 'json'}"}
        while True:
            try:
                resp = requests.post(url, text, params=params).text
                content=json.loads(resp)
                #print('finish parse:',re.sub('\s+',' ',content['sentences'][0]['parse'].replace('\n',' ')))
                return re.sub('\s+',' ',content['sentences'][0]['parse'].replace('\n',' '))
            except ConnectionRefusedError:
                print('error, retrying...')
    def clean(self,sentence):
        initial=''
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in self.verbtags:
                    vbflag=1
                else:
                    vbflag=0
            else:
                node=re.match('([^\)]+)(\)*)',tag.strip())
                if node:
                    if vbflag==1:
                        initial+=' ('+node.group(1)+')'
                    else:
                        initial+=' '+node.group(1)
        return initial
    def cleanclear(self,sentence):
        initial=''
        for tag in sentence.split():
            if tag[0]=='(':
                if tag[1:] in self.verbtags:
                    vbflag=1
                else:
                    vbflag=0
            else:
                node=re.match('([^\)]+)(\)*)',tag.strip())
                if node:
                    initial+=' '+node.group(1)
        return initial
    def work(self,a):
        url = 'http://166.111.139.15:9000'
        params = {'properties' : r"{'annotators': 'tokenize,ssplit,pos,lemma,parse,depparse', 'outputFormat': 'json'}"}
        while True:
            try:
                resp = requests.post(url,a,params=params).text
                break
            except ConnectionRefusedError:
                print('error, retrying...')
        #print(resp)
        #input()
        try:
            content=json.loads(resp)
        except:
            print('resperror: ',resp)
            return 1
        for sentence in content['sentences']:
            for i in sentence['enhancedPlusPlusDependencies']:
                if 'nsubjpass' in i.values():
                    return 1
              #  else:
              #      print('notin:\n',i)
        return 0
        

    def __init__(self,\
                patchlength=3,\
                maxlength=700,\
                embedding_size=100,\
                num_verbs=2,\
                allinclude=False,\
                shorten=False,\
                shorten_front=False,\
                testflag=False,\
                passnum=0):   #几句前文是否shorten #是否输出不带tag,只有单词的句子 

#patchlength:每次输入前文额外的句子的数量.
#maxlength:每句话的最大长度.(包括前文额外句子).超过该长度的句子会被丢弃.
#embedding_size:词向量维度数.
        self.url = 'http://166.111.139.15:9000'
        self.shorten=shorten
        self.shorten_front=shorten_front   #几句前文是否shorten #是否输出不带tag,只有单词的句子 
        self.patchlength=patchlength
        self.maxlength=maxlength
        self.embedding_size=embedding_size
        self.num_verbs=num_verbs
        self.allinclude=allinclude
        self.passnum=passnum
        print('pas',passnum)
        self.verbtags=['VB','VBZ','VBP','VBD','VBN','VBG'] #所有动词的tag
        self.model=word2vec.load('train/combine100.bin')   #加载词向量模型
        print('loaded model')
        self.oldqueue=Queue()
        self.testflag=testflag

        self.testfile=open('input.txt')
        if testflag==False:
            self.resp=open(r'train/resp2').readlines()
            self.readlength=len(self.resp)
            print('readlength',self.readlength)
            self.pointer=random.randint(0,self.readlength-1)
#            self.pointer=1621919
            print('pointer',self.pointer)
            for _ in range(self.patchlength):
                self.oldqueue.put(self.resp[self.pointer])
                self.pointer+=1
        else:
            for _ in range(self.patchlength):
                if shorten_front==True:
                    #self.oldqueue.put(input('0:type sentence:'))
                    self.oldqueue.put(self.testfile.readline())
                else:
                    #self.oldqueue.put(self.parse(input('0:type sentence:')))
                    self.oldqueue.put(self.parse(self.testfile.readline()))

#加载文字

#加载原型词典(把动词变为它的原型)
        with open('train/ldict2', 'rb') as f:
            self.ldict = pickle.load(f)
        with open('train/tagdict', 'rb') as f:
            self.tagdict = pickle.load(f)
        
        print('loaded lemma')



    def lemma(self,verb):
        if verb in self.ldict:
            return self.ldict[verb]
        else:
            params = {'properties' : r"{'annotators': 'lemma', 'outputFormat': 'json'}"}
            resp = requests.post(self.url, verb, params=params).text
            content=json.loads(resp)
            word=content['sentences'][0]['tokens'][0]['lemma']
            self.ldict[verb]=word
            print('errorverb: ',verb,word)
            return word

    def list_tags(self,batch_size):
        while True:#防止读到末尾
            inputs=[]
            pads=[]
            answer=[]
            count=0
            while len(answer)<batch_size:
                #print('batch_size',batch_size)

                if self.testflag==True:
                    if self.shorten==True:
                        #sentence=input('1:')
                        sentence=self.testfile.readline()
                    else:
                        #print('parsed!')
                        #sentence=self.parse(input('1:'))
                        sentence=self.parse(self.testfile.readline())
                else:
                    sentence=self.resp[self.pointer]
                    if len(sentence)>20000:
                        print('pointer',self.pointer)
                        raise MemoryError
                    self.pointer+=1
                    #print(self.pointer)
                    if self.pointer==self.readlength:
                        self.pointer=0
                        print('epoch')

                outword=[]
                total=0
#前文句子
                newqueue=Queue()
                for _ in range(self.patchlength):
                    oldsentence=self.oldqueue.get()
                    newqueue.put(oldsentence)
                    for tag in oldsentence.split():
                        if tag[0]=='(':
                            if tag not in self.tagdict:
                                self.tagdict[tag]=len(self.tagdict)
                                print(len(self.tagdict))
                            tagword=[0]*self.embedding_size
                            tagword[self.tagdict[tag]]=1
                            if not self.shorten_front:
                                outword.append(tagword)
                        else:                
                            node=re.match('([^\)]+)(\)*)',tag.strip())
                            if node:
                                #group(1) 单词
                                if node.group(1) in self.model:
                                    outword.append(self.model[node.group(1)].tolist())
                                else:
                                    outword.append([0]*self.embedding_size)
                                #group(2) 括号
                                if not self.shorten_front:
                                    tagword=[0]*self.embedding_size
                                    tagword[0]=1
                                    for _ in range(len(node.group(2))-1):
                                        outword.append(tagword)
                self.oldqueue=newqueue
                self.oldqueue.put(sentence)
                self.oldqueue.get()
                #print('point at:',self.resp.tell())

#本句                
                for tag in sentence.split():
                    if tag[0]=='(':
#去除情态动词
                        if tag=='(MD':
                            mdflag=1
                        else:
                            mdflag=0
                            if tag[1:] in self.verbtags:
                                tag='(VB'
                                vbflag=1
                            else:
                                vbflag=0
                            if tag not in self.tagdict:
                                self.tagdict[tag]=len(self.tagdict)
                                print(len(self.tagdict))
                            tagword=[0]*self.embedding_size
                            tagword[self.tagdict[tag]]=1
                            if not self.shorten:
                                outword.append(tagword)
                    else:
                        if mdflag==0:
                            node=re.match('([^\)]+)(\)*)',tag.strip())
                            if node:
                                if node.group(1) in self.model:
                                    if vbflag==1:
#去除时态
                                        node2=self.lemma(node.group(1))
                                        if node2 in self.model:
                                            if not (node2=='is' or node2=='have'):#去除is,have

                                                outword.append(self.model[node2].tolist())
                                        else:
                                            outword.append([0]*self.embedding_size)
                                    else:
                                        outword.append(self.model[node.group(1)].tolist())
                                else:
                                    outword.append([0]*self.embedding_size)
                                if not self.shorten:
                                    tagword=[0]*self.embedding_size
                                    tagword[0]=1
                                    for _ in range(len(node.group(2))-1):
                                        outword.append(tagword)
                outword=np.array(outword)
#句子过长
                if outword.shape[0]>self.maxlength:
                    continue
                temp=[0,0]
                qq=self.work(self.cleanclear(sentence))
                temp[qq]=1
        #        print(self.clean(sentence),qq)
        #        input()
                answer.append(temp)
#补零
                pads.append(outword.shape[0])
                outword=np.pad(outword,((0,self.maxlength-outword.shape[0]),(0,0)),'constant')
                inputs.append(outword)

            inputs=np.array(inputs)
#构建输出
#用完整个输入,从头开始
#continue the 'while True' loop
            return inputs,pads,answer

if __name__ == '__main__':
    model = reader()
    for i in range(2):
        model.list_tags(1)
