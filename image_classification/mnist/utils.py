# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:59:21 2020

@author: arun
"""


def loss_fn(inference_fn, inputs, labels):
    # Using sparse_softmax cross entropy
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=inference_fn(inputs), labels=labels
        )
    )


def accuracy_fn(inference_fn, inputs, labels):

    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def getpreds(inference_fn, inputs, labels):

    prediction = tf.nn.softmax(inference_fn(inputs))

    return tf.argmax(prediction, 1), tf.argmax(labels, 1)
