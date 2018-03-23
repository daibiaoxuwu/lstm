#encoding:utf-8
#xuni.py 关于虚拟语气


import numpy as np
import word2vec
import re
import os
import pickle
from queue import Queue

class reader(object):
    def __init__(self,\
                patchlength=3,\
                maxlength=700,\
                embedding_size=100,\
                num_verbs=2,\
                allinclude=False):

#patchlength:每次输入前文额外的句子的数量.
#maxlength:每句话的最大长度.(包括前文额外句子).超过该长度的句子会被丢弃.
#embedding_size:词向量维度数.
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


    def list_tags(self):
        count=0
        while True:#防止读到末尾
            inputs=[]
            pads=[]
            answer=[]
            while True:
                count+=1
                print('c:',count)
                sentence=self.resp.readline()
                if sentence=='':
                    self.resp.seek(0, os.SEEK_SET)
                    sentence=self.resp.readline()
                    print('epoch')

#剔除句子
                '''
                total=0 #统计本句话有几个动词
#筛选只有一个动词的句子                
                for tag in sentence.split():
                    if tag[0]=='(':
                        if tag[1:] in self.verbtags:
                            total+=1
                if (self.allinclude==True and total<self.num_verbs) or (self.allinclude==False and total!=self.num_verbs):
                    self.oldqueue.put(sentence)
                    self.oldqueue.get()
                    continue
                '''
#前文句子
                outword=[]
                singleverb=0 #在翻译成数字的时候的动词计数器
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
                self.oldqueue=newqueue
                self.oldqueue.put(sentence)
                self.oldqueue.get()
                #print('point at:',self.resp.tell())

#本句                
                printline=0
                realline=''
                for tag in sentence.split():
                    if tag[0]=='(':
#去除情态动词
                        if tag=='(MD':
                            mdflag=1
                          #  printline=1
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
                            outword.append(tagword)
                    else:
                        if True:
                            node=re.match('([^\)]+)(\)*)',tag.strip())
                            if node:
                                if node.group(1)=='would':
                                    printline=1
                                realline+=node.group(1)
                                realline+=' '
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
                if printline==1:
                    print(realline)
                    input()
                outword=np.array(outword)
#句子过长
                if outword.shape[0]>self.maxlength:
                    print('pass')
                    answer=answer[:-1]
                    continue
#补零

#continue the 'while True' loop

if __name__ == '__main__':
    model = reader()
    model.list_tags()
