#!/usr/bin/env python2.7
#-*- encoding: utf-8 -*-


import pandas as pd
import numpy as np
import tensorflow as tf


def matrix():
    """
    main function
    """
    df = pd.read_csv('/home/ao/Documents/Projet/pokemon//data/full.csv')

    mapping = {'NoT':0, 'False':0, 'True':1}
    count = 1
    for name in df['Type 10'].unique():
        mapping[name] = count
        count +=1

    df['Legendary1']=df['Legendary1'].astype('int64')
    df['Legendary0']=df['Legendary0'].astype('int64')
    df =df.replace({'Type 10': mapping, 'Type 20': mapping, 'Type 11': mapping,
                    'Type 21': mapping})

    df = df.drop(['First_pokemon', 'Second_pokemon', 'Name1','Name0'], axis=1)

    array = df.as_matrix()
    features = array[:,0:-1]
    target = np.asarray([array[:,-1]]).reshape(100000, 1)

    return features, target




def network(features, target):


    trainf = features[:80000,:]
    traint = target[:80000,:]

    testf = features[80000:,:]
    testt = target[80000:,:]

    tf_trainf = tf.placeholder(tf.float32, shape=[None, 20],
                               name='TrainFeaturs')
    tf_traint = tf.placeholder(tf.float32, shape=[None, 1], name='TrainTarget')


    #tf_testf = tf.placeholder(tf.float32, shape=[None, 20])
    #tf_testt = tf.placeholder(tf.float32, shape=[None, 1])


    # First part
    w1 = tf.Variable(tf.random.normal([20,4]), name='Weight1')
    b1 = tf.Variable(tf.zeros([4]), name='Bias1')

    z1 = tf.matmul(tf_trainf, w1) + b1
    a1 = tf.nn.sigmoid(z1)

    # Second part

    w2 = tf.Variable(tf.random.normal([4,1]), name='Weight2')
    b2 = tf.Variable(tf.zeros([1]), name='Bias2')

    z2 = tf.matmul(a1, w2) + b2
    py = tf.nn.sigmoid(z2)

    ###############
    cost = tf.reduce_mean(tf.square(py - tf_traint))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)

    correct = tf.equal(tf.round(py), tf_traint)
    
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    ##############
    #z1 = tf.matmul(tf_testf, w1) + b1
    #a1 = tf.nn.sigmoid(z1)
    #z2 = tf.matmul(a1, w2) + b2
    #py = tf.nn.sigmoid(z2)
    #val_cor = tf.equal(tf.round(py), tf_testt)
    #val_acc = tf.reduce_mean(tf.cast(val_cor, tf.float32))

    sess = tf.Session()
    writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch in range(10000):
        sess.run(train, feed_dict={tf_trainf: trainf,
                                   tf_traint: traint})

    writer.close()


    print "Accuracy = ", sess.run(accuracy, feed_dict={tf_trainf: trainf,
                                                       tf_traint: traint})

    #print "Accuracy TEST = ", sess.run(val_acc, feed_dict={tf_testf: testf,
    #                                                        tf_testt: testt})

def main():
    features, target = matrix()
    network(features, target)

if __name__ =='__main__':
    main()

