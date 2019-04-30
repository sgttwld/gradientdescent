"""
Example of a sampling based gradient descent (ascent) implementation in tensorflow to find the 
parameters of a probability distribution that maximizes the expected value of a given utility function
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/gradientdescent
"""

import numpy as np
import tensorflow as tf
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# parameters
numSamples = 1000
lr = .01
precision = 1e-7
numEp = 20000

## placeholders
x = tf.placeholder(tf.float64, shape=[None])

## utility function
U = -(x**2-2)**2/4 + 1

## variables
theta = tf.Variable(np.ones(2),dtype=tf.float64)
mu = tf.exp(theta[0])
sigma = tf.exp(theta[1])

## parametrized (gaussian) density evaluated at x
p = tf.exp(-(x-mu)**2/(2*sigma**2))/tf.sqrt(2*math.pi*sigma**2)

## objective for gradient descent 
## (logp makes the auto-derivative correct in case of sampling) 
obj = tf.reduce_mean(tf.log(p)*U)

## true objective (only for evaluation)
obj_true = tf.reduce_mean(U)

## optimizer and training operator
optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=.9, beta2=.999,epsilon=1e-08,name='Adam')
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(-obj)

## initializer
init = tf.global_variables_initializer()
   
   
with tf.Session() as sess:
 
    ## initializing
    sess.run(init)
    obj0 = 0
        
    for n in range(0,numEp):
    
        ## data generation    
        mu_curr = mu.eval(session=sess)
        sigma_curr = sigma.eval(session=sess)
        data = np.random.normal(loc=mu_curr, scale=sigma_curr, size=numSamples)
        
        ## run gradient descent
        sess.run(train_op,feed_dict={x: data})
        
        ## evaluation
        obj_curr = obj_true.eval(session=sess,feed_dict={x: data})       
        if n % 50 == 0:
            print('ep:',n,'f(t) = ', obj_curr)
        if abs(obj_curr-obj0) < precision:
            break
        else:
            obj0 = obj_curr

    mu_fin = mu.eval(session=sess)
    sigma_fin = sigma.eval(session=sess)


# discretizing the distribution for visualization
xx = np.linspace(0,2,50)
dens = np.exp(-(xx-mu_fin)**2/(2*sigma_fin**2))/np.sqrt(2*math.pi*sigma_fin**2)

## for command line visualization:
import hipsterplot as hplt

def show(v,stretch=2):
    hplt.plot(np.concatenate([[val]*stretch for val in v]), num_x_chars=stretch*len(v),num_y_chars=10)

print('utility')
show(-(xx**2-2)**2 / 4 + 1,stretch=1)

print('probability density', ' mu =',mu_fin, ' sigma =',sigma_fin)
show(dens,stretch=1)