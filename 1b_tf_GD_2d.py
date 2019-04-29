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
t = tf.Variable(tf.random_normal((1,)))

# objective
obj = (t**2-2)**2

# optimizer and training operator
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(obj)

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
        t_curr = t.eval(session=sess)
        print('ep',n,'t = ', t_curr[0],', f(t) = ', obj_curr[0])
        if abs(obj_curr-obj0) < precision:
            break
        else:
            obj0 = obj_curr