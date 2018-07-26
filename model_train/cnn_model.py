# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:42:16 2018

@author: Administrator
"""
import tensorflow as tf


class CNN_TEXT(object):
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders of input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 不同的卷积核创造不同的卷积池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv_pool-{}".format(filter_size)):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.a_pool = tf.concat(pooled_outputs, 3)
        self.a_pool_shape = tf.reshape(self.a_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.a_drop = tf.nn.dropout(self.a_pool_shape, self.dropout_keep_prob)

        '''
        # 添加三个隐层
        num_filters_up = num_filters_total
        fc_input = self.a_drop
        for i,fc_size in enumerate([1000,500,128]):
            with tf.name_scope('FC-{}'.format(fc_size)):
                W_fc = tf.Variable(tf.truncated_normal([num_filters_up,fc_size],stddev=0.1,name = 'W_fc'))
                b_fc = tf.Variable(tf.constant(0.1, shape=[fc_size]), name="b_fc")
                l2_loss += tf.nn.l2_loss(W_fc)
                l2_loss += tf.nn.l2_loss(b_fc)
                fc_out = tf.nn.relu(tf.nn.xw_plus_b(fc_input, W_fc, b_fc, name="fc_out"))
                fc_input = fc_out
                num_filters_up = fc_size
        self.fc_out = fc_out
        num_filters_fc = num_filters_up
        '''

        # output and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.a_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accurancy')

