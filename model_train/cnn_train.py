# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:02:03 2018

@author: Administrator
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import datetime
from tensorflow.contrib import learn
from data_process import data_process_train
from model_train.cnn_model import CNN_TEXT

# Parameters
#### ==================================================####

# data parameters
tf.flags.DEFINE_float("dev_sample_percentage", 0.025, "验证集大小")
tf.flags.DEFINE_string("datapath", "../data/train_texts.csv", "数据路径")

# model hyper parameters
tf.flags.DEFINE_integer("embedding_dim", 200, "词向量维度")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "滤波器的大小")
tf.flags.DEFINE_integer("num_filters", 128, "信道数量")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("evaluate_every", 100, "每训练多少步以后在验证集上进行训练")
tf.flags.DEFINE_integer("checkpoint_every", 100, "每次训多少步以后断电存储模型")
tf.flags.DEFINE_integer("num_checkpoints", 5, "所能容纳的最多的模型数量")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr, value))
print("")

###===========================1 Data_process===============###
# 导入数据
print('loaing data:\n')
x_text, y = data_process_train.load_data_and_labels(FLAGS.datapath)
x_text = x_text.astype(np.str_)
print('训练数据的大小为{}\n'.format(len(y)))

# 建立词表，将词数字化
def f1(x):
    return len(x)
# 取文档平均词个数的25%为文档训练所截取的词个数
max_document_len = int(x_text.apply(f1).mean()/4)
print('文档取词个数为：{}\n'.format(max_document_len))
vocab_process = learn.preprocessing.VocabularyProcessor(max_document_len)
x = vocab_process.fit_transform(x_text)
x = np.array(list(x))
print('词表建立，数字化结束！\n')

# shuffle data
np.random.seed(123)
shuffle_index = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_index]
y_shuffled = y[shuffle_index]

# split train/dev set
from sklearn.cross_validation import train_test_split
x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled,
                                                test_size=FLAGS.dev_sample_percentage, random_state=0)

# 释放内存
del x_shuffled, y_shuffled, x, y, x_text, attr, value

print("Vocabulary Size: {:d}".format(len(vocab_process.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

print('一切准备就绪，我要就开始训练了！！！')
### =========================2 train ========================###
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN_TEXT(
            sequence_length=max_document_len,
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_process.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # 训练（优化器）
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        grads_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_vars, global_step=global_step)

        # 跟踪梯度值的变化并且保存
        grad_summaries = []
        for g, v in grads_vars:
            if g is not None:
                grad_hist = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_scalar = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist)
                grad_summaries.append(sparsity_scalar)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 输出路径创建
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
        print("Writing to {}\n".format(out_dir))

        # 跟踪loss以及accuaacy的变化
        loss_sum = tf.summary.scalar('loss', cnn.loss)
        accuracy_sum = tf.summary.scalar('accuracy', cnn.accuracy)

        # train_summaries
        train_op_sum = tf.summary.merge([loss_sum, accuracy_sum, grad_summaries_merged])
        train_sum_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_sum_dir, sess.graph)

        # dev_summarier
        dev_op_sum = tf.summary.merge([loss_sum, accuracy_sum])
        dev_sum_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_sum_dir, sess.graph)

        # Checkpoint directory creat.
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 保存词表
        vocab_process.save(os.path.join(out_dir, 'vocab'))

        # 初始化
        sess.run(tf.global_variables_initializer())


        # 一次训练
        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_op_sum, cnn.loss, cnn.accuracy],
                feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat().replace('T', '-').split('.')[0]
            print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        # 一次验证
        def dev_step(x_batch, y_batch, writer=None):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0}
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, dev_op_sum, cnn.loss, cnn.accuracy],
                feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat().replace('T', '-').split('.')[0]
            print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # 恢复保存的神经网络结构，实现断点续训
        ckpt = tf.train.get_checkpoint_state(os.path.join(out_dir, 'checkpoints'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('之前的模型导入成功！')
            
        # 循环进行训练多个epoch
        for epoch in range(FLAGS.num_epochs):
            print('正在进行第 {} 次循环！'.format(epoch+1))
            # 产生bache数据集
            batches = data_process_train.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size)
            for batche in batches:
                x_batch, y_batch = zip(*batche)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\n=========Evaluation=========:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("\n")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            print('第 {} 次循环结束！'.format(epoch + 1))
