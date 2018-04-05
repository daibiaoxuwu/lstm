#encoding:utf-8
#run3.py 只看含有两个动词的句子

import numpy as np
import tensorflow as tf
import time
import os
import sys, getopt
import importlib
from elapsed import elapsed





os.environ["CUDA_VISIBLE_DEVICES"]=""#环境变量：使用第一块gpu
logs_path = 'log/run'
saving_path='ckpt/run/run.ckpt'
load_path='ckpt/run/'

#神经网络的输入是一句只有一个动词的句子（以及其语法树），把动词变为原型，语法树的tag变为了VB。
#并预测它的动词时态。如果它不为0，输入变为这句话以及他前面的patchlength句话。
#语法树结构：（VB love）会被变为三个标签：（VB的（100维）one-hot标签，love的词向量标签，反括号对应的全0标签。
#每个反括号对应一个单独的标签，而正括号没有。

patchlength=3                   #输入的前文句子的数量
embedding_size=100              #词向量维度数量
maxlength=700                   #输入序列最大长度
initial_training_rate=0.00005     #学习率
training_iters = 10000000       #迭代次数
batch_size=50                   #batch数量
display_step = 1               #多少步输出一次结果
saving_step=20                 #多少步保存一次
num_verbs=2                     #一次看两个动词
allinclude=False                #只看刚好含有num_verbs个动词的句子
passnum=0

time_verbose_flag=False         #测量输入和运行的时间比

reader = importlib.import_module('readertest')
rnnmodel = importlib.import_module('rnnmodel')


config=tf.ConfigProto()
loadold=False
shorten=False
shorten_front=False
testflag=False
try:
    opts, args = getopt.getopt(sys.argv[1:],"hg:lp:x:n:r:m:ais:oStP:")
except getopt.GetoptError:
    print('使用不正确.详见python run.py -h')
    sys.exit()
for opt, arg in opts:
    if opt == '-h':
        print('''usage:
run.py  -g 使用gpu号(0,1) 默认:0    
        -l limit 是否限制gpu显存为50%(不填)(默认:不限制)
        -p patchlength前文数量(数字,默认:3)
        -x maxlength句子长度(数字,默认:700)
        -n num_verbs单词数量(数字,默认:2)
        -r 读取模型(模型名,文件名去掉.py 默认:reader)
        -m rnn模型(模型名,文件名去掉.py 默认:rnnmodel)
        -a 存储的allow_growth(不填)(默认:不允许)
        -s 保存用的标识,默认:run.log路径为log/run,保存路径为ckpt/run/run.ckpt
        -i allinclude 默认为读入时只读入含有num_verbs个动词的句子, 设置后读入所有含有不少于num_verbs的句子
        -o 是否从上次的模型加载
        -S shorten=True shorten_front=True
        -P 读入时跳过几个
        -t test
        ''')
        sys.exit()
    elif opt=="-g":
        os.environ["CUDA_VISIBLE_DEVICES"]=arg
    elif opt=="-l":
        config.gpu_options.per_process_gpu_memory_fraction=0.4#占用40%显存
    elif opt=="-p":
        patchlength=int(arg)
    elif opt=="-x":
        maxlength=int(arg)
    elif opt=="-n":
        num_verbs=int(arg)
    elif opt=="-r":
        reader = importlib.import_module(arg)
    elif opt=="-m":
        rnnmodel = importlib.import_module(arg)
    elif opt=="-a":
        tf_config.gpu_options.allow_growth = True
    elif opt=="-s":
        logs_path = 'log/'+arg
        saving_path='ckpt/'+arg+'/'+arg+'.ckpt'
        load_path='ckpt/'+arg
    elif opt=="-i":
        allinclude=True
    elif opt=="-o":
        loadold=True
    elif opt=="-S":
        shorten=True
        shorten_front=True
    elif opt=='-t':
        batch_size=1
        saving_step=100000000000

        reader = importlib.import_module('readertest')
#        config.device_count={'gpu':0}#使用cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        testflag=True
    elif opt=='-P':
        passnum=int(arg)



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
            allinclude=False,\
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
print('start session')
with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())#初始化变量
    if loadold:
        ckpt = tf.train.get_checkpoint_state(load_path)
        saver.restore(session, ckpt.model_checkpoint_path)  
    step = 0
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
        print('i')
        print('b',batch_size)
        inputs,pads,answers=data.list_tags(batch_size)
        print('0')
#运行一次
        _,pred, acc, loss, onehot_pred, summary= session.run([model.optimizer, model.pred, model.accuracy, model.cost, model.pred, merged], \
                                                feed_dict={model.x: inputs, model.y: answers, model.p:pads})
#累加计算平均正确率
        if testflag==True:
            print(pred)
            print(tf.argmax(pred[0]).eval())
            print(type(tf.argmax(pred[0]).eval()))
            print('pred:', data.printtag(tf.argmax(pred[0]).eval()))
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
                start_time=time.time()
                acc_total = 0
                loss_total = 0
#保存
            if step % saving_step ==0:
                print('saved to: ', saver.save(session,saving_path,global_step=step))
    print("Optimization Finished!")
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
