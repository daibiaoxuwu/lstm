#encoding:utf-8

import numpy as np
import tensorflow as tf
import time
import os
import reader
import rnnmodel

os.environ["CUDA_VISIBLE_DEVICES"]="0"#环境变量：使用第一块gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1"#环境变量：使用第一块gpu
logs_path = 'log/runmul'
saving_path='/home/djl/runmul/n.ckpt'

patchlength=0
patchlength=3
#神经网络的输入是一句只有一个动词的句子（以及其语法树），把动词变为原型，语法树的tag变为了VB。
#并预测它的动词时态。如果它不为0，输入变为这句话以及他前面的patchlength句话。
#语法树结构：（VB love）会被变为三个标签：（VB的（100维）one-hot标签，love的词向量标签，反括号对应的全0标签。
#每个反括号对应一个单独的标签，而正括号没有。
#one-hot的意思就是，假如有

embedding_size=100
maxlength=200
maxlength=700
initial_training_rate=0.001
training_iters = 10000000
batch_size=50
display_step = 20
saving_step=2000


start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

data=reader.reader(patchlength=patchlength,\
                    maxlength=maxlength,\
                    embedding_size=embedding_size)
count,inputs,pads,answers=data.list_tags(3,batch_size)

# Target log path
writer = tf.summary.FileWriter(logs_path)

model=rnnmodel.rnnmodel(vocab_size=6,\
                        maxlength=maxlength,\
                        embedding_size=embedding_size,\
                        initial_training_rate=initial_training_rate,\
                        batch_size=batch_size)

saver=tf.train.Saver()
# Launch the graph
print('start session')
#config=tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction=0.4
#with tf.Session(config=config) as session:
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    step = 0
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)
    count=patchlength
    while step < training_iters:
        count,inputs,pads,answers=data.list_tags(count,batch_size)
        _, acc, loss, onehot_pred= session.run([model.optimizer, model.accuracy, model.cost, model.pred], \
                                                feed_dict={model.x: inputs, model.y: answers, model.p:pads})
        loss_total += loss
        acc_total += acc
        step += 1
        model.global_step += 1
        #print(model.global_step.eval())
        if step % display_step == 0:
            print("Iter= " + str(step+1) + ", used: "+str(count)+ ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step)," Elapsed time: ", elapsed(time.time() - start_time))
            start_time=time.time()
            acc_total = 0
            loss_total = 0
        if step % saving_step ==0:
            print('saved to: ', saver.save(session,saving_path,global_step=step))
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
