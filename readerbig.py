#encoding:utf-8
#reader.py 只看含有一个动词的句子(十分之一左右)

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import word2vec
import re
import pickle
from queue import Queue

class reader(object):
    def __init__(self,\
                patchlength=0,\
                maxlength=200,\
                embedding_size=100):

#patchlength:每次输入前文额外的句子的数量.
#maxlength:每句话的最大长度.(包括前文额外句子).超过该长度的句子会被丢弃.
#embedding_size:词向量维度数.
        self.patchlength=patchlength
        self.maxlength=maxlength
        self.embedding_size=embedding_size
        self.verbtags=['VB','VBZ','VBP','VBD','VBN','VBG'] #所有动词的tag
        self.model=word2vec.load('train/combine100.bin')   #加载词向量模型
        print('loaded model')
        self.oldqueue=Queue()
        self.resp=open(r'train/resp')
        for _ in range(self.patchlength):
            self.oldqueue.put(self.resp.readline())


#加载文字

#加载原型词典(把动词变为它的原型)
        with open('train/lemma2', 'rb') as f:
            self.ldict = pickle.load(f)
        print('loaded lemma')
    def lemma(self,verb):
        if verb in self.ldict:
            return self.ldict[verb]
        else:
            print('errverb:',verb)
            return verb


    def list_tags(self,batch_size):
        with open(r'train/resp') as resp:
            while True:#防止读到末尾
                realength=0
                tagdict={')':0}
                inputs=[]
                pads=[]
                answer=[]
                while len(answer)<batch_size:
                    sentence=resp.readline()
                    if sentence=='':
                        resp.seek(0, os.SEEK_SET)
                        sentence=resp.readline()
                        print('epoch')
                    self.oldqueue.put(sentence)

                    outword=[]
                    total=0
#筛选只有一个动词的句子                
                    for tag in sentence.split():
                        if tag[0]=='(':
                            if tag[1:] in self.verbtags:
                                total+=1
                    if total!=1:
                        self.oldqueue.get()
                        continue
#前文句子
                    newqueue=Queue()
                    for _ in range(self.patchlength):
                        oldsentence=self.oldqueue.get()
                        newqueue.put(oldsentence)
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
                    newqueue.put(self.oldqueue.get())
                    oldqueue=newqueue
                    oldqueue.get()

#本句                
                    for tag in sentence.split():
                        if tag[0]=='(':
#去除情态动词
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
#去除时态
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
#句子过长
                    if outword.shape[0]>self.maxlength:
                        print('pass')
                        answer=answer[:-1]
                        continue
#补零
                    pads.append(outword.shape[0])
                    outword=np.pad(outword,((0,self.maxlength-outword.shape[0]),(0,0)),'constant')
                    inputs.append(outword)

                inputs=np.array(inputs)
#构建输出
                answers=np.zeros((len(answer),len(self.verbtags)))
                for num in range(len(answer)):
                    answers[num][answer[num]]=1
#用完整个输入,从头开始
#continue the 'while True' loop
                else:
                    return inputs,pads,answers

if __name__ == '__main__':
    model = reader().list_tags(5)
