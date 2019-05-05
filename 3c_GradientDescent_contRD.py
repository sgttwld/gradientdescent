"""
Example of a gradient descent (ascent) implementation in tensorflow to find the parameters of
a continuous probability distribution that solves the rate distortion problem
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/gradientdescent
"""

import numpy as np
import tensorflow as tf
import math, os

def get_density(x,mu,sigma):
    return tf.exp(-(x-mu)**2/(2*sigma**2))/tf.sqrt(2*math.pi*sigma**2)

def get_utility(x,mu,sigma):
    return tf.exp(-(x-mu)**2/(2*sigma**2))

## parameters: problem
beta = 20
numW = 10
sigma_U = 0.3

## parameters: algorithm
numEp = 10000
lr = .001
precision = 1e-7
numSamples = 250

## placeholder
W = tf.placeholder(tf.int64, shape=[None,1])
X = tf.placeholder(tf.float64, shape=[None,1])

## constants:
pw_vec = np.ones(numW)/numW
pw = tf.constant(pw_vec)
mu_U = tf.constant(np.linspace(0,1,numW))

## utility
U = get_utility(X,mu=tf.gather(mu_U,indices=W),sigma=sigma_U)

## variables
theta = tf.Variable(np.ones(numW),dtype=tf.float64)
alpha = tf.Variable(np.linspace(-4,4,numW),dtype=tf.float64)

## parametrization
sigma = tf.exp(theta)
mu = 1/(1+tf.exp(-alpha))

## density
mu_D = tf.gather(mu,indices=W)
sigma_D = tf.gather(sigma,indices=W)
fxgw = get_density(X,mu=mu_D,sigma=sigma_D)

## optimal prior
fx_w = tf.stack([get_density(X,mu[i],sigma[i]) for i in range(numW)])
fx = tf.reduce_mean(fx_w,axis=0)

## objective
EU_true = tf.reduce_mean(U)
obj = tf.reduce_mean(tf.log(fxgw)*(U-tf.log(fxgw/fx)/(2*beta) - 1/beta))
obj_true = tf.reduce_mean(U-tf.log(fxgw/fx)/beta)

#### TENSORBOARD/1 ####
obj_summary = tf.summary.scalar('objective', obj)
obj_true_summary = tf.summary.scalar('free energy', obj_true)
EU_summary = tf.summary.scalar('expected utility', EU_true)
merged = tf.summary.merge_all()
#######################

## optimizer and training operator
#optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.999,epsilon=1e-08,name='Adam')
optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=.9, beta2=.999,epsilon=1e-08,name='Adam')
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(-obj)

## initializer
init = tf.global_variables_initializer()

## running the TF session
with tf.Session() as sess:

    #### TENSORBOARD/2 ####
    run=len(os.listdir(path='./logs'))
    writer = tf.summary.FileWriter('./logs/r{}'.format(run), sess.graph)
    #######################

    ## initializing
    sess.run(init)
    obj0 = 0    
    p0 = 0

    for n in range(0,numEp):
        
        ## data generation    
        mu_curr = mu.eval(session=sess)
        sigma_curr = sigma.eval(session=sess)
        W_D = []
        X_D = []
        for k in range(numSamples):
            w = np.random.choice(range(numW), p = pw_vec)
            x = np.random.normal(loc=mu_curr[w], scale=sigma_curr[w], size=1)
            W_D.append([w])
            X_D.append(x) 
            
        ## run gradient descent
        sess.run(train_op,feed_dict={W: W_D, X: X_D})
        
        #### TENSORBOARD/3 ####
        summary = sess.run(merged,feed_dict={W: W_D, X: X_D})
        writer.add_summary(summary,n)
        #######################

        ## evaluation
        obj_curr = obj_true.eval(session=sess,feed_dict={W: W_D, X: X_D}) 
        EU_curr = EU_true.eval(session=sess,feed_dict={W: W_D, X: X_D})      
        
        if n % 10 == 0:
            print('ep:',n,'E_p[U] = ', EU_curr, 'F[p] = ', obj_curr)
        if abs(obj_curr-obj0) < precision:
            break
        else:
            obj0 = obj_curr


## visualization of learning with tensorboard: 
## console command: tensorboard --logdir="./logs" --port 6006

## visualization of resulting density with pyplot
import matplotlib.pyplot as plt
a = np.linspace(0,1,100)
p0 = np.vstack([np.exp(-(a-mu_curr[i])**2/(2*sigma_curr[i]**2)) for i in range(numW)])
p = np.einsum('ik,i->ik',p0,1.0/np.einsum('ik->i',p0))
plt.pcolor(p)
plt.show()

