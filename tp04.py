import tensorflow as tf  
import numpy as np  
  
  
w = tf.Variable(tf.random_normal([1], -1, 1))  
b = tf.Variable(tf.zeros([1]))  
  
isTrain = True
#isTrain = False
train_steps = 100  
checkpoint_steps = 50  
checkpoint_dir = '/home/djl/tmp/'  
  
saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b  
  
with tf.Session() as sess:  
    if isTrain:  
        for i in range(train_steps):  
            sess.run(train, feed_dict={x: x_data})  
            if (i + 1) % checkpoint_steps == 0:  
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i+1)  
    else:  
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  
        print(ckpt)
        print('a')
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path)  
        else:  
            print('e')
    print(sess.run(w))  
    print(sess.run(b))  
