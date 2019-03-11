#!/usr/bin/env python2.7
#-*- encoding: utf-8 -*-

from datetime import datetime
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def matrix():
    """
    main function
    """
    df = pd.read_csv(sys.argv[-1])

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

    # separate train and test dataset
    trainf, testf, traint, testt = train_test_split(features, target,
                                                    test_size=0.20,
                                                    random_state=11)


    tf_features = tf.placeholder(tf.float32, shape=[None, 20],
                               name='TrainFeaturs')
    tf_targets = tf.placeholder(tf.float32, shape=[None, 1],
                               name='TrainTarget')


    # First part
    with tf.name_scope('FirstLayer'):
        w1 = tf.Variable(tf.random.normal([20,10]), name='Weight1')
        w1_hist = tf.summary.histogram('w1', w1)
        b1 = tf.Variable(tf.zeros([10]), name='Bias1')
        b1_hist = tf.summary.histogram('b1', b1)
    
        with tf.name_scope('Preactivation1'):
            z1 = tf.matmul(tf_features, w1) + b1
        with tf.name_scope('Activation1'):
            a1 = tf.nn.sigmoid(z1)

    # Second part
    with tf.name_scope('SecondLayer'):
        w2 = tf.Variable(tf.random.normal([10,5]), name='Weight2')
        w2_hist = tf.summary.histogram('w2', w2)
        b2 = tf.Variable(tf.zeros([5]), name='Bias2')
        b2_hist = tf.summary.histogram('b2', b2)
    
        with tf.name_scope('Preactivation2'):
            z2 = tf.matmul(a1, w2) + b2
        with tf.name_scope('Activation2'):
            a2 = tf.nn.sigmoid(z2)

    # Third part
    with tf.name_scope('ThirdLayer'):
        w3 = tf.Variable(tf.random.normal([5,1]), name='Weight3')
        w3_hist = tf.summary.histogram('w3', w3)
        b3 = tf.Variable(tf.zeros([1]), name='Bias3')
        b3_hist = tf.summary.histogram('b3', b3)
    
        with tf.name_scope('Preactivation3'):
            z3 = tf.matmul(a2, w3) + b3
        with tf.name_scope('Activation3'):
            py = tf.nn.sigmoid(z3)

    # Train Part
    with tf.name_scope('Cost'):
        cost = tf.reduce_mean(tf.square(py - tf_targets))
    cost_sum = tf.summary.scalar(name='Cost', tensor=cost)

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    # Result Parut
    with tf.name_scope('Accuracy'):
        correct = tf.equal(tf.round(py), tf_targets)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    ac_sum = tf.summary.scalar(name='Accuracy', tensor=accuracy)

    # Summary
    merged_summary_op = tf.summary.merge_all()
    # Exec
    with tf.Session() as sess:
        logdir = "/tmp/tensorflow/graph/train10_5_1/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        writer = tf.summary.FileWriter(logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch in range(10000):
            summary = sess.run([optimizer, cost, accuracy, merged_summary_op],
                                     feed_dict={tf_features: trainf,
                                                tf_targets: traint})
            writer.add_summary(summary[-1], epoch)

        writer.close()

        print
        print "Accuracy on Train = ", sess.run(accuracy,
                                               feed_dict={tf_features: trainf,
                                                          tf_targets: traint})

        print "Accuracy on Test = ", sess.run(accuracy,
                                              feed_dict={tf_features: testf,
                                                         tf_targets: testt})

def main():
    features, target = matrix()
    network(features, target)

if __name__ =='__main__':
    main()

