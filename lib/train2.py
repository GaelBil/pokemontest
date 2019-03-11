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
    target = list()
    for value in array[:,-1]:
        if value == 1:
            hot_encode = [0,1]
        else:
            hot_encode= [1,0]
        target.append(hot_encode)
    target = np.asarray(target)

    return features, target





def network(features, target):

    # separate train and test dataset
    trainf, testf, traint, testt = train_test_split(features, target,
                                                    test_size=0.20,
                                                    random_state=11)


    tf_features = tf.placeholder(tf.float32, shape=[None, 20],
                               name='TrainFeaturs')
    tf_targets = tf.placeholder(tf.float32, shape=[None, 2],
                               name='TrainTarget')


    # First part
    with tf.name_scope('FirstLayer'):
        w1 = tf.Variable(tf.random.normal([20,2]), name='Weight1')
        w1_hist = tf.summary.histogram('w1', w1)
        b1 = tf.Variable(tf.zeros([2]), name='Bias1')
        b1_hist = tf.summary.histogram('b1', b1)

        with tf.name_scope('Preactivation1'):
            z1 = tf.matmul(tf_features, w1) + b1
        with tf.name_scope('Activation1'):
            py = tf.nn.softmax(z1)
    

    # Train Part
    with tf.name_scope('Cost'):
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=tf_targets,
                                                       logits=z1)
#    cost_sum = tf.summary.scalar(name='Cost', tensor=cost)

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    # Result Parut
    with tf.name_scope('Accuracy'):
        correct = tf.equal(tf.argmax(py, 1), tf.argmax(tf_targets, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    ac_sum = tf.summary.scalar(name='Accuracy', tensor=accuracy)

    # Summary
    merged_summary_op = tf.summary.merge_all()
    # Exec
    with tf.Session() as sess:
        logdir = "/tmp/tensorflow/graph/train2/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
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

