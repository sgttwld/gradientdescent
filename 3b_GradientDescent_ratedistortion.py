"""
Example of a gradient descent (ascent) implementation in tensorflow to find the discrete 
probability distribution that solves the rate distortion problem
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/gradientdescent
"""

import numpy as np
import tensorflow as tf

## parameters: problem
beta = 20.0
sgm = .5
N = 50
numW = 10


## parameters: algorithm
numEp = 5000
lr = .1
precision = 1e-10

## constants:
a = tf.constant(np.linspace(0,1,N))
mus = tf.constant(np.linspace(0,1,numW))
U = tf.stack([tf.exp(-(a-mus[i])**2/(2*sgm**2)) for i in range(numW)])
pw = tf.constant(np.ones(numW)/numW)

## variables
theta = tf.Variable(np.ones((numW,N)),dtype=tf.float64)
gamma = tf.Variable(np.ones(N),dtype=tf.float64)

## parametrizations
pa = tf.exp(gamma)/tf.reduce_sum(tf.exp(gamma)) 
Z = tf.einsum('ik->i',tf.exp(theta))
pagw = tf.einsum('i,ik->ik',1.0/Z,tf.exp(theta))

## objective
EU = tf.einsum('i,ik->',pw,pagw*U)    
DKL = tf.einsum('i,ik->',pw,pagw*tf.log(tf.einsum('ik,k->ik',pagw,1.0/pa)))
obj = EU - DKL/beta

## optimizer and training operator
optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=.9, beta2=.999,epsilon=1e-08,name='Adam')
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(-obj)

## initializer
init = tf.global_variables_initializer()

## running the TF session
with tf.Session() as sess:
    
    ## initializing
    sess.run(init)
    obj0 = 0    
    p0 = 0
        
    for n in range(0,numEp):
        
        ## run gradient descent
        sess.run(train_op)
        
        ## evaluation
        obj_curr = obj.eval(session=sess)
        if n % 500 == 0:
            print('ep:',n,'f(t) = ', obj_curr)
        if abs(obj_curr-obj0) < precision:
            print('ep:',n,'f(t) = ', obj_curr)
            break
        else:
            obj0 = obj_curr
        
    prob = pagw.eval(session=sess)
    Umat = U.eval(session=sess)


## visualize result
import matplotlib.pyplot as plt
plt.pcolor(prob)
plt.show()

