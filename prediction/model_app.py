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
from data_process import data_process_train
from prediction import predictionapp



# ==================================== Parameters ============================== #
# SQL Parameters
tf.flags.DEFINE_string("user", "root", "数据库密码")
tf.flags.DEFINE_string("passwd", "123456", "数据库密码")
tf.flags.DEFINE_integer("port", 3306, "数据库端口")
tf.flags.DEFINE_string("host", "localhost", "数据库地址")
tf.flags.DEFINE_string("dbname", 'test', "数据库名称")
tf.flags.DEFINE_string("input_table","test_news", "导入的数据库表")
tf.flags.DEFINE_string("output_table","test_news_predict", "新的数据库表")
tf.flags.DEFINE_integer("start", 0, "取出数据的记录开始行")
tf.flags.DEFINE_integer("nums", 50, "一共取出多少条记录")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_string("checkpoint_dir","../model_train/runs128", "训练的模型存储路径")
tf.flags.DEFINE_boolean("save_bool", False, "是否存入新表")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr, value))
print("全部参数打印完成！\n")


# ========================================== 导入自己的测试数据 =============================#
x_raw,save_data = predictionapp.get_sql_data1(FLAGS.input_table,FLAGS.user,FLAGS.passwd,
                                              FLAGS.host,FLAGS.port,FLAGS.dbname,FLAGS.nums,FLAGS.start)

# 将原始数据映射到词典
vocab_path = os.path.join(FLAGS.checkpoint_dir,"vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print("\nEvaluating...\n")


#### ========================================   计算测试 ===================================###
# checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.checkpoint_dir, "checkpoints"))
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 同时加载计算图以及变量
        ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint_dir, "checkpoints"))
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint_file = ckpt.model_checkpoint_path
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # 找到我们需要进行计算的变量，目标输出变量
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            # 开始收集预测结果
            if FLAGS.nums <= 2000:# 一次性处理记录不多时，直接处理
                all_predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
            else:# 数据量过多，用bache分解数据
                batches = data_process_train.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
                # 收集所有的预测输出
                all_predictions = []
                for x_test_batch in batches:
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
        else:
            print('模型不存在，请检查模型路径或者训练模型！\n')

# 对应类别名字典,prediction转化为标签
new_dict = [open('../data_process/new_class.txt').read()]
new_dict = eval(new_dict[0])
labels = []
for i in all_predictions:
    labels.append(new_dict[int(i)])

# 粘合数据
save_data['prediction'] = pd.Series(all_predictions)
save_data['label'] = pd.Series(labels)
print('开始存入数据库！')
if FLAGS.save_bool:
    '''存入一张新表中，包含原始的信息'''
    predictionapp.insert_new(save_data,FLAGS.output_table,FLAGS.user,FLAGS.passwd,FLAGS.host,FLAGS.port,FLAGS.dbname)
else:
    '''存入旧表中'''
    predictionapp.insert_raw(all_predictions,labels,save_data,FLAGS.input_table,FLAGS.user,FLAGS.passwd,FLAGS.host,FLAGS.port,FLAGS.dbname)




