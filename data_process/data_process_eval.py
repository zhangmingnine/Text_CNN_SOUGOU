#!-*-coding:utf-8 -*-

from data_process.data_process_train import *

# 1、用户输入单个新闻文本文件
def data_sigle(inputfile):
    with open(inputfile, errors='ignore') as f:
        content = f.read()
        content = clean_text(content)
        content = seg_text(content)
    return [content,None]


# 2、用户输入含有多条新闻文本的文件，独立输入新闻文本的csv文件。
def data_eval(inputfile):
    f = open(inputfile, encoding='gbk')
    data = pd.read_csv(f)
    text = data['content'].apply(clean_text)
    text = text.apply(seg_text)
    return text



# 3、测试模型的数据导入，即数据库的数据导入
def data_test(inputfile):
    f = open(inputfile, encoding='gbk')
    data = pd.read_csv(f)
    text1 = data['title'].apply(clean_text)
    text2 = data['description'].apply(clean_text)
    text3 = data['content'].apply(clean_text)
    # title,description,content连接到一起
    text = text1 + text2 + text3
    text = text.apply(seg_text)
    return text

