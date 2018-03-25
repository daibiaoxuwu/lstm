#encoding:utf-8
#reader.py 只看含有一个动词的句子(十分之一左右)

import numpy as np
import word2vec
import re
import os
import pickle
import random
from queue import Queue

def getMem(ini):
    with open('/proc/meminfo') as f:
        total = int(f.readline().split()[1])
        free = int(f.readline().split()[1])
        buffers = int(f.readline().split()[1])
        cache = int(f.readline().split()[1])
        while(buffers<1000000):
            print('wait',buffers)
            time.sleep(60)
            buffers = int(f.readline().split()[1])
        return buffers

class reader(object):
    def __init__(self,\
                patchlength=3,\
                maxlength=700,\
                embedding_size=100,\
                num_verbs=2,\
                allinclude=False,\
                shorten=False,\
                shorten_front=False):   #几句前文是否shorten #是否输出不带tag,只有单词的句子 

#patchlength:每次输入前文额外的句子的数量.
#maxlength:每句话的最大长度.(包括前文额外句子).超过该长度的句子会被丢弃.
#embedding_size:词向量维度数.
        self.shorten=shorten
        self.shorten_front=shorten_front   #几句前文是否shorten #是否输出不带tag,只有单词的句子 
        self.patchlength=patchlength
        self.maxlength=maxlength
        self.embedding_size=embedding_size
        self.num_verbs=num_verbs
        self.allinclude=allinclude
        self.verbtags=['VB','VBZ','VBP','VBD','VBN','VBG'] #所有动词的tag
        self.model=word2vec.load('train/combine100.bin')   #加载词向量模型
        self.tagdict={')':0}
        print('loaded model')
        self.oldqueue=Queue()
        self.resp=open(r'train/resp').readlines()
        self.readlength=len(self.resp)
        self.pointer=random.randint(0,self.readlength-1)
        for _ in range(self.patchlength):
            self.oldqueue.put(self.resp[self.pointer])
            self.pointer+=1


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
        while True:#防止读到末尾
            inputs=[]
            pads=[]
            answer=[]
            count=0
            while len(answer)<batch_size:
                getMem(0)

                sentence=self.resp[self.pointer]
                self.pointer+=1
                if self.pointer==self.readlength:
                    self.pointer=0
                    sentence=self.resp[self.pointer]
                    self.pointer+=1
                    print('epoch')

                outword=[]
                total=0
                singleverb=0
                getMem(1)
#筛选只有一个动词的句子                
                for tag in sentence.split():
                    if tag[0]=='(':
                        if tag[1:] in self.verbtags:
                            total+=1
                if (self.allinclude==True and total<self.num_verbs) or (self.allinclude==False and total!=self.num_verbs):
                    self.oldqueue.put(sentence)
                    self.oldqueue.get()
                    continue
#前文句子
                newqueue=Queue()
                getMem(2)
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
                getMem(3)
                self.oldqueue=newqueue
                self.oldqueue.put(sentence)
                self.oldqueue.get()
                getMem(4)
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
                                if singleverb==0:
                                    answer.append(self.verbtags.index(tag[1:]))
                                    singleverb=1
                                elif singleverb<self.num_verbs:
                                    answer[-1]*=len(self.verbtags)
                                    answer[-1]+=self.verbtags.index(tag[1:])
                                    singleverb+=1
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
                getMem(5)
                outword=np.array(outword)
                getMem(6)
#句子过长
                if outword.shape[0]>self.maxlength:
#                    print('pass')
                    answer=answer[:-1]
                    continue
#补零
                getMem(7)
                pads.append(outword.shape[0])
                outword=np.pad(outword,((0,self.maxlength-outword.shape[0]),(0,0)),'constant')
                inputs.append(outword)
                getMem(8)

            inputs=np.array(inputs)
            getMem(9)
#构建输出
            answers=np.zeros((len(answer),pow(len(self.verbtags),self.num_verbs)))
            for num in range(len(answer)):
                answers[num][answer[num]]=1
#用完整个输入,从头开始
#continue the 'while True' loop
            return inputs,pads,answers

if __name__ == '__main__':
    model = reader()
    for i in range(500000000):
        model.list_tags(500)
        print('i',i)
