#encoding:utf-8

import numpy as np
import tensorflow as tf
import time
import os
from readerbig import reader
from rnnmodel import rnnmodel

os.environ["CUDA_VISIBLE_DEVICES"]="0"#环境变量：使用第一块gpu
logs_path = 'log/run'
saving_path='ckpt/run/run.ckpt'

#神经网络的输入是一句只有一个动词的句子（以及其语法树），把动词变为原型，语法树的tag变为了VB。
#并预测它的动词时态。如果它不为0，输入变为这句话以及他前面的patchlength句话。
#语法树结构：（VB love）会被变为三个标签：（VB的（100维）one-hot标签，love的词向量标签，反括号对应的全0标签。
#每个反括号对应一个单独的标签，而正括号没有。

patchlength=0                   #输入的前文句子的数量
embedding_size=100              #词向量维度数量
maxlength=200                   #输入序列最大长度
initial_training_rate=0.001     #学习率
training_iters = 10000000       #迭代次数
batch_size=50                   #batch数量
display_step = 50               #多少步输出一次结果
saving_step=2000                #多少步保存一次


start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

#input
data=reader(patchlength=patchlength,\
            maxlength=maxlength,\
            embedding_size=embedding_size)

#通过在命令行运行tensorboard --logdir=$logs_path 然后按提示在浏览器打开http://....:6006即可.
#使用远程服务器的话如果想在自己的电脑上看,需要找到某个公开的端口,如8001, 运行tensorboard --logdir=log/rnn3 然后浏览器打开http://xxx.xxx.xxx.xxx:8001即可(xxx为服务器域名)
writer = tf.summary.FileWriter(logs_path)

model=rnnmodel(vocab_size=6,\
            maxlength=maxlength,\
            embedding_size=embedding_size,\
            initial_training_rate=initial_training_rate,\
            batch_size=batch_size)

# 数据存储器.一定要写在整个网络的末尾.
saver=tf.train.Saver()
tf.summary.scalar('train_loss', model.cost)
tf.summary.scalar('accuracy', model.accuracy)
merged = tf.summary.merge_all()

# Launch the graph
print('start session')
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4#占用40%显存
with tf.Session(config=config) as session:
#with tf.Session() as session:
    session.run(tf.global_variables_initializer())#初始化变量
    step = 0
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)
    while step < training_iters:
#读入一个batch的数据
#重用的话只要实现自己的reader.py就行.
#inputs:batch_size个输入句子,形状为[batch_size, maxlength, embedding_size]
#pads:batch内每句话的长度,形状为[batch_size]
#answers:输入的答案,形状为[batch_size,vocab_size]
        inputs,pads,answers=data.list_tags(batch_size)
#运行一次
        _, acc, loss, onehot_pred, summary= session.run([model.optimizer, model.accuracy, model.cost, model.pred, merged], \
                                                feed_dict={model.x: inputs, model.y: answers, model.p:pads})
#累加计算平均正确率
        loss_total += loss
        acc_total += acc
        step += 1
#帮global_step(用来调节学习率指数下降的)加一
        model.global_step += 1
        #print(model.global_step.eval())
#输出
        if step % display_step == 0:
            writer.add_summary(summary, step)
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step)," Elapsed time: ", elapsed(time.time() - start_time))
            start_time=time.time()
            acc_total = 0
            loss_total = 0
#保存
        if step % saving_step ==0:
            print('saved to: ', saver.save(session,saving_path,global_step=step))
    print("Optimization Finished!")
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
