# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:52:53 2018

@author: Administrator
"""

import re       
import os
import pandas as pd
import pymysql

''' 
1、训练数据的初步清洗以及分类存储
'''
threh = 30 #字符数小于这个数目的content将不被保存
# 对应字典
dicurl = {'auto.sohu.com': '汽车', 'it.sohu.com': '互联网', 'health.sohu.com': '健康',
          'sports.sohu.com': '体育', 'travel.sohu.com': '旅游', 'learning.sohu.com': '教育',
          'career.sohu.com': '招聘', 'cul.sohu.com': '文化', 'mil.news.sohu.com': '军事',
          'house.sohu.com': '房产', 'yule.sohu.com': '娱乐', 'women.sohu.com': '时尚',
          'media.sohu.com': '传媒', 'news.sohu.com': '其他', '2008.sohu.com': '奥运',
          'business.sohu.com': '商业'}

"""全角转半角"""
def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

# 对每个语料提取出类别url和内容content以及title
def data_clean(datapath,con_path,title_path):
    listdir = os.listdir(datapath)
    for path in listdir:
        print('正在处理 {} 文件'.format(path))
        filename = os.path.join(datapath,path)
        file = open(filename, 'r',encoding='gb18030').read()
        #正则匹配出url和content
        patternURL = re.compile(r'<url>(.*?)</url>', re.S)
        patternTitle = re.compile(r'<contenttitle>(.*?)</contenttitle>', re.S)
        patternCtt = re.compile(r'<content>(.*?)</content>', re.S)
    
        classes = patternURL.findall(file)
        titles = patternTitle.findall(file)
        contents = patternCtt.findall(file)
        # 把所有内容小于30字符的文本全部过滤掉
        for i in list(range(contents.__len__()))[::-1]:
            if len(contents[i]) < threh or len(titles[i]) == 0:
                contents.pop(i)
                classes.pop(i)
                titles.pop(i)
        # 把URL进一步提取出来，只提取出一级url作为类别
        for i in range(classes.__len__()):
            patternClass = re.compile(r'http://(.*?)/',re.S)
            classi = patternClass.findall(classes[i])
            classes[i] = classi[0]
        # 按照RUL作为类别保存到samples文件夹中
        for i in range(classes.__len__()):
            #contents[i] = contents[i].replace('\ue40c','')
            #contents[i] = strQ2B(contents[i])
            titles[i] = titles[i].replace('\ue40c', '')
            titles[i] = strQ2B(titles[i])
            if classes[i] in dicurl:
                #file1 = con_path +'/'+ dicurl[classes[i]] + '.txt'
                file2 = title_path +'/'+  dicurl[classes[i]] + '.txt'
                #with open(file1,'a+',encoding = 'utf8') as f1:
                    #f1.write(contents[i]+'\n')   #加\n换行显示
                with open(file2,'a+',encoding = 'utf8') as f2:
                    f2.write(titles[i]+'\n')   #加\n换行显示


'''
2、测试数据，数据库来源的数据获取

# 打开数据库连接
db = pymysql.connect(host = '124.202.155.72',
                     port = 33063,
                     user = "newsreadonly",
                     passwd = "newsreadonly",
                     db = "news_db",
                     charset = 'utf8')

# cat = [line.strip() for line in open('类别.txt',encoding = 'utf-8-sig').readlines()]
'''

def test_data():
    sqlcmd = "SELECT url,title,description,content,category FROM t_news_detail limit 10;"
    data = pd.read_sql(sqlcmd, db)
    if not os.path.exists('../data'):
        os.makedirs('../data')
    data.to_csv('../data/test/test_data.csv',index=False)



if __name__ == '__main__':
    #test_data()
    data_clean('../data/reduced','../data/contents','../data/titles')

                
