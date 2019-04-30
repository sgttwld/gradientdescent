"""
Example of a gradient descent (ascent) implementation in tensorflow to find the discrete probability 
distribution that maximizes the expected value of a given function
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/gradientdescent
"""

import numpy as np
import tensorflow as tf

# parameters
numEp = 50000
lr = .1
precision = 1e-6
N = 50

# constants
t = tf.constant(np.linspace(0,2,N))
U = -(t**2-2)**2/4 + 1

# variables
theta = tf.Variable(np.ones(N),dtype=tf.float64)

# parametrization of p(t)
p = tf.exp(theta)/tf.reduce_sum(tf.exp(theta))

# objective
obj = tf.reduce_sum(p*U)    # expected utility E[U] = sum_x p(x) U(x)

# optimizer and training operator
optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=.9, beta2=.999,epsilon=1e-08,name='Adam')
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(-obj)

# initializer
init = tf.global_variables_initializer()

# running the TF session
with tf.Session() as sess:
    
    ## initializing
    sess.run(init)
    obj0 = 0
            
    for n in range(0,numEp):
        
        ## run gradient descent
        sess.run(train_op)
        
        ## evaluation
        obj_curr = obj.eval(session=sess)
        if n % 50 == 0:
            print('ep:',n,'E[U] = ', obj_curr)
        if abs(obj_curr-obj0) < precision:
            break
        else:
            obj0 = obj_curr

    prob = p.eval(session=sess)
    tmax = t.eval(session=sess)[np.argmax(prob)]

print('argmax p(t) =',tmax)

# for command line visualization:
import hipsterplot as hplt
def show(v,stretch=2):
    hplt.plot(np.concatenate([[val]*stretch for val in v]), num_x_chars=stretch*len(v),num_y_chars=10)

print('utility')
show(-(np.linspace(0,2,N)**2-2)**2 / 4 + 1,stretch=1)

print('probability distribution')
show(prob,stretch=1)