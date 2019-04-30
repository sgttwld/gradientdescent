"""
Example of gradient descent to find the minimum of a function using tensorflow
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/gradientdescent
"""

import numpy as np
import tensorflow as tf

# parameters
numEp = 1000
lr = .1
precision = 1e-10

# variables
xy = tf.Variable(tf.random_normal((2,)))

# objective
obj = tf.exp(-(xy[0]**2+xy[1]**2-2)**2)

# optimizer and training operator
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(-obj)

# initializer
init = tf.global_variables_initializer()

# running the TF session
with tf.Session() as sess:
    
    ## initializing
    sess.run(init)
    obj0 = 0
    
    for n in range(numEp):
        
        ## run gradient descent
        sess.run(train_op)
        
        ## evaluation
        obj_curr = obj.eval(session=sess)
        xy_curr = xy.eval(session=sess)
        print('ep',n,'(x,y)=', xy_curr,'radius=',np.linalg.norm(xy_curr),'f(x,y)=', obj_curr)
        if abs(obj_curr-obj0) < precision:
            break
        else:
            obj0 = obj_curr