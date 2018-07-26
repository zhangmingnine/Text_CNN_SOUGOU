# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 09:28:59 2018

@author: Administrator
"""
import re       
import os
import jieba
import pandas as pd
import numpy as np
import random


# 1 清洗文本，去除标点符号数字以及特殊符号
def clean_text(content):
    text = re.sub(r'[+——！，；／·。？、~@#￥%……&*“”《》：（）［］【】〔〕]+', '', content)
    text = re.sub(r'[▲!"#$%&\'()*+,-./:;<=>\\?@[\\]^_`{|}~]+', '', text)
    text = re.sub('\d+', '', text)
    text = re.sub('\s+', '', text)
    return text

# 2 利用jieba包进行分词，并并且去掉停词，返回分词后的文本
def seg_text(text):
    stop = [line.strip() for line in open('stopwords.txt', encoding='utf8').readlines()]
    text_seged = jieba.cut(text.strip())
    outstr = ''
    for word in text_seged:
        if word not in stop:
            outstr += word
            outstr += " "
    return outstr.strip()


# 3 批量处理文本数据，清洗，分词，去停用词以及最后保存文本数据和对应标签为csv文件。
def savefile(path, savepath):
    listdirs = os.listdir(path)
    count = 0
    result = pd.DataFrame(columns=['description', 'labels'])
    newclass = {}  # 数字标签和文本类别
    for dirs in listdirs:
        print('\n开始处理 {} 文件,一共{}个文件,这是第{}个文件'.format(dirs,len(listdirs),count+1))
        newclass[count] = dirs.split('.')[0]
        fullname = os.path.join(path, dirs)
        f = open(fullname, encoding='utf8')
        data = f.read()
        contentlists = data.split('\n')
        if len(contentlists)>20000:
            contentlists = pd.Series(contentlists[:20000])
        elif len(contentlists)< 5000:
            contentlists = pd.Series(random.choices(contentlists,k = 5000))
        else:
            contentlists = pd.Series(contentlists)
        texts = contentlists.apply(clean_text)
        texts = texts.apply(seg_text)
        labels = [count] * len(texts)
        labels = pd.Series(labels)
        tt = pd.concat([texts, labels], axis=1)
        tt.columns = ['description', 'labels']
        result = result.append(tt)
        print('\n已经处理好了 {} 文件,这是第{}个文件'.format(dirs, count+1))
        count += 1
    with open('new_class.txt', 'w') as f:
        f.write(str(newclass))
    result.to_csv(savepath, index=False)  # 保存分词好的文件


# 将数字标签转为one_hot向量
def make_one_hot(data1):
    return (np.arange(14) == data1[:, None]).astype(np.int32)


# 4 加载为适合的数据类型
def load_data_and_labels(datapath):
    """
    将数据分割为文本句子和标签labels两部分，并且格式化为需要的格式，返回数据
    """
    # Load data from files
    data = pd.read_csv(datapath, encoding='gbk')
    data = data.dropna(axis=0,how='any')
    # Split by words and labels
    x_text = data['description']
    labels = data['labels']
    # Generate one_hot vector
    labels = np.asarray(labels)
    y = make_one_hot(labels)
    return [x_text, y]


# 5、生成batch数据集
def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    raw_datapath = '../data/contents'
    save_datapath = '../data/train_texts.csv'
    savefile(raw_datapath,save_datapath)





