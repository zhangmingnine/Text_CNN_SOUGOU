# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:49:37 2018

@author: Administrator
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.contrib import learn
from data_process import data_process_eval,data_process_train
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("inputfile", "../data/test/test_data.csv", "需要预测的数据文件名")
tf.flags.DEFINE_string("outputfile", "..", "保存预测文件的位置，直接在上层目录保存")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_string("checkpoint_dir","../model_train/runs", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "是否计算所有的数据")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr, value))
print("snlili")


# 导入自己的测试数据，可以选择针对整个文件计算还是单独的一个文本
wen = input('输入的是原始文本还是文本文件？如果是原始文本请输入‘True’')
if wen is True:
    x_raw = input('输入你的原始文本')
else:
    wen1 = input('输入的是单个文本的文件，还是多条文本的txt文件？如果是单条文本，请输入‘True’')
    if wen1 is True:
        x_raw = data_process_eval.data_sigle(FLAGS.inputfile)
    else:
        x_raw = data_process_eval.data_eval(FLAGS.inputfile)


# 将原始数据映射到词典
vocab_path = os.path.join(FLAGS.checkpoint_dir,"vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

#### ========================= 计算测试 ======================###
# checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.checkpoint_dir, "checkpoints"))
ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint_dir, "checkpoints"))
checkpoint_file = ckpt.model_checkpoint_path
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 同时加载计算图以及变量
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 根据名字从图中找到对应的placeholders，不需要input_y,只需要输入x
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # 找到我们需要进行计算的变量，目标输出变量
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # 产生bache只有一个循环一次
        batches = data_process_train.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # 收集所有的预测输出
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# 对应类别名字典
new_dict = [open('../data_process/new_class.txt').read()]
new_dict = eval(new_dict[0])
labels = []
for i in all_predictions:
    labels.append(new_dict[int(i)])

# 保存结果到一个csv中,读入原来的文件在添加新的一列预测结果
savedata = pd.read_csv(FLAGS.inputfile,encoding='gbk')
'''
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions, np.array(labels)))
out_path = os.path.join(FLAGS.outputfile, "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w', newline='') as f:
    cf = csv.writer(f)
    cf.writerows(predictions_human_readable)
'''
savedata['prediction'] = pd.Series(all_predictions)
savedata['labels'] = pd.Series(labels)
savedata1 = savedata.iloc[:500,:]
savedata1.to_csv(FLAGS.outputfile+'/'+'zmshiyan.csv',index=False)


