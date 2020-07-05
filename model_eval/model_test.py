# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:49:37 2018
dfg
@author: Administrator
"""
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
from data_process import data_process_eval,data_process_train
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("inputfile", "../data/test_data.csv", "Data source for the  data.")
tf.flags.DEFINE_string("outputfile", "..", "保存预测文件的位置，直接在上层目录保存")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_string("checkpoint_dir", "../model_train/runs", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "是否计算所有的数据")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr, value))
print("")

# 导入自己的测试数据，可以选择针对整个文件计算还是单独的一个文本
if FLAGS.eval_train:
    x_raw = data_process_eval.data_test(FLAGS.inputfile)
    y_test = None
else:
    x_raw = [
        "日前 数据 显示 道口 贷 平台 累计 成交额 正式 突破 亿元 服务 实体 企业 超过 家 企业 融资 成本 之间 项目 零 逾期 年 道口 贷 单个 月度 撮合 成交额 超过 亿元 去年 同期相比 增长 呈 高 增长 态势 截止 月 日 道口 贷已 家 核心 ",
        "三个 月 激烈 角逐 月 日 京东 集团 发起 举办 中国 新闻出版 研究院 豆瓣 协办 首届 京东 文学奖 在京举行 颁奖典礼 揭晓 获奖 名单 春风 乡村 生活 图景 六部 文学 佳作 最终 胜出 摘得 年度 京东 文学奖 百万 奖金 莫言 周国平 蒋方舟 史航 方文山 到场 当晚 颁奖典礼 "]
    y_test = [0, 1]

# 将原始数据映射到词典
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
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

# 如果知道真实标签，让我们来看一看对比的结果如何吧
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}%".format(correct_predictions / float(len(y_test)) * 100))


# 对应类别名字典
new_dict = [open('../data_process/new_class.txt').read()]
new_dict = eval(new_dict[0])
labels = []
for i in all_predictions:
    labels.append(new_dict[int(i)])

# 保存结果到一个csv中,文本，类别，标签（三个）
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions, np.array(labels)))

out_path = os.path.join(FLAGS.outputfile, "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w', newline='') as f:
    cf = csv.writer(f)
    cf.writerows(predictions_human_readable)

