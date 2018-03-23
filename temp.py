#encoding:utf-8
#reader.py 只看含有一个动词的句子(十分之一左右)

import numpy as np
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
        self.tagdict={')':0}
        print('loaded model')
        self.oldqueue=Queue()
        self.resp=open(r'temp')
        for _ in range(self.patchlength):
            self.oldqueue.put(self.resp.readline())


#加载文字

    def list_tags(self,batch_size):
        while True:#防止读到末尾
            inputs=[]
            pads=[]
            answer=[]
            while len(answer)<batch_size:
                sentence=self.resp.readline()
                if sentence=='':
                    self.resp.seek(0, os.SEEK_SET)
                    sentence=self.resp.readline()
                    print('epoch')
                print(sentence)


if __name__ == '__main__':
    model = reader()
    model.list_tags(5)
    model.list_tags(5)
