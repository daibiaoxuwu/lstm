#encoding:utf-8
#run3.py 只看含有两个动词的句子

import numpy as np
import tensorflow as tf
import time
import os
import sys, getopt
import importlib
from elapsed import elapsed




def check(verse):
    content=verse.split('.','\n')

os.environ["CUDA_VISIBLE_DEVICES"]=""#环境变量：使用第一块gpu
logs_path = 'log/run'
saving_path='ckpt2/run/run.ckpt'
load_path='ckpt2/run/'

#神经网络的输入是一句只有一个动词的句子（以及其语法树），把动词变为原型，语法树的tag变为了VB。
#并预测它的动词时态。如果它不为0，输入变为这句话以及他前面的patchlength句话。
#语法树结构：（VB love）会被变为三个标签：（VB的（100维）one-hot标签，love的词向量标签，反括号对应的全0标签。
#每个反括号对应一个单独的标签，而正括号没有。

patchlength=3                   #输入的前文句子的数量
embedding_size=100              #词向量维度数量
maxlength=700                   #输入序列最大长度
initial_training_rate=0.0001     #学习率
training_iters = 10000000       #迭代次数
batch_size=50                   #batch数量
display_step = 20               #多少步输出一次结果
saving_step=1000                  #多少步保存一次
num_verbs=1                     #一次看两个动词
allinclude=True                #只看刚好含有num_verbs个动词的句子
passnum=0

time_verbose_flag=False         #测量输入和运行的时间比

reader = importlib.import_module('reader')
rnnmodel = importlib.import_module('rnnmodel')


config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.45#占用45%显存
loadold=True
shorten=False
shorten_front=False
testflag=False
multiflag=False
multinum=1
tf_config.gpu_options.allow_growth = True

        testflag=True
        loadold=True
        allinclude=True
        batch_size=1
        saving_step=100000000000
        display_step=10000000000

#        config.device_count={'gpu':0}#使用cpu
        #os.environ["CUDA_VISIBLE_DEVICES"] = ""#>?????????????????

        multiflag=True
        multinum=4



start_time = time.time()
def getMem():
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

#input
data=reader.reader(patchlength=patchlength,\
            maxlength=maxlength,\
            embedding_size=embedding_size,\
            num_verbs=num_verbs,\
            allinclude=allinclude,\
            shorten=shorten,\
            shorten_front=shorten_front,\
            testflag=testflag,\
            passnum=passnum)


model=rnnmodel.rnnmodel(vocab_single=6,\
            maxlength=maxlength,\
            embedding_size=embedding_size,\
            initial_training_rate=initial_training_rate,\
            batch_size=batch_size,\
            num_verbs=num_verbs)

#通过在命令行运行tensorboard --logdir=$logs_path 然后按提示在浏览器打开http://....:6006即可.
#使用远程服务器的话如果想在自己的电脑上看,需要找到某个公开的端口,如8001, 运行tensorboard --logdir=log/rnn3 然后浏览器打开http://xxx.xxx.xxx.xxx:8001即可(xxx为服务器域名)
writer = tf.summary.FileWriter(logs_path)

# 数据存储器.一定要写在整个网络的末尾.
saver=tf.train.Saver()
tf.summary.scalar('train_loss', model.cost)
tf.summary.scalar('accuracy', model.accuracy)
merged = tf.summary.merge_all()

# Launch the graph
max_acc_total=0
print('start session')

#multi-time test
while True:
    print('iter')
    multitime=0
    while True:
        print('nut,',multitime)

        with tf.Session(config=config) as session:
            session.run(tf.global_variables_initializer())#初始化变量
            if loadold:
#reload the test model multiple times
                if multiflag==True:
                    ckpt = tf.train.get_checkpoint_state('ckpt2/p'+str(multitime)+'n1')
                    print('p'+str(multitime)+'n1')
                else:
                    ckpt = tf.train.get_checkpoint_state(load_path)

                saver.restore(session, ckpt.model_checkpoint_path)
            step = 1
            acc_total = 0
            loss_total = 0

            writer.add_graph(session.graph)
            while step < training_iters:
#读入一个batch的数据
#重用的话只要实现自己的reader.py就行.
#输出:count:指针,指向读到文件的哪个位置
#inputs:batch_size个输入句子,形状为[batch_size, maxlength, embedding_size]
#pads:batch内每句话的长度,形状为[batch_size]
#answers:输入的答案,形状为[batch_size,vocab_size]
                #print('i')
                #print('b',batch_size)
                if multitime==0:
                    #print('listtags')

                    inputs,pads,answers,singleverb=data.list_tags(batch_size)
                    #print('sv',singleverb)
#运行一次
                _,pred, acc, loss, onehot_pred, summary= session.run([model.optimizer, model.pred, model.accuracy, model.cost, model.pred, merged], \
                                                        feed_dict={model.x: inputs, model.y: answers, model.p:pads})
#累加计算平均正确率
                if testflag==True:
                    #print(pred)
                    #print(tf.argmax(pred[0]).eval())
                    #print(type(tf.argmax(pred[0]).eval()))
                    print('pred:', data.printtag(tf.argmax(pred[0]).eval()))
                    step+=1
                else:
                    loss_total += loss
                    acc_total += acc
                    step += 1
#帮global_step(用来调节学习率指数下降的)加一
                    model.global_step += 1
                    #print(model.global_step.eval())
#输出
                    if step % display_step == 0:
                        writer.add_summary(summary, step)
                        print('free memory= '+str(int(getMem()/1000000))+"GB, Iter= " + str(step+1) + ", Average Loss= " + \
                              "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                              "{:.2f}%".format(100*acc_total/display_step)," Elapsed time: ", elapsed(time.time() - start_time))
                        if testflag==False and acc_total>max_acc_total:
                            max_acc_total=acc_total
                            print('saved to: ', saver.save(session,saving_path3,global_step=step))
                        start_time=time.time()
                        acc_total = 0
                        loss_total = 0
#保存
                    if step % saving_step ==0:
                        print('saved to: ', saver.save(session,saving_path,global_step=step))
            if testflag==True:
                multitime+=1
                if multitime>=singleverb:
                    print('mts',multitime,singleverb)
                    break
