# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:53:52 2018

@author: Administrator
"""
import pymysql
import pandas as pd
from sqlalchemy import create_engine
from data_process.data_process_train import seg_text,clean_text


'''
#存储数据（试验）
data = pd.read_csv('../data/test/test_data100.csv',encoding = 'gbk')
yconnect = create_engine('mysql+pymysql://root:123456@localhost:3306/test?charset=utf8')  
pd.io.sql.to_sql(data,'test_news', yconnect, schema='test', if_exists='replace',index = False)  
'''


### 打开数据库连接
def link_sql(user,password,host,port,dbname):
    conn = pymysql.connect(host=host,
                         port=port,
                         user=user,
                         passwd=password,
                         db=dbname,
                         charset='utf8')
    return conn

# 1、从数据库中取出数据,取出sta+1条记录的n条数据
def get_sql_data1(tabelname,user,password,host,port,dbname,n,sta):
    db = link_sql(user,password,host,port,dbname)
    sqlcmd = "SELECT * FROM {} limit {} offset {};".format(tabelname,n,sta)
    data = pd.read_sql(sqlcmd, db)
    text1 = data['title'].apply(clean_text)
    text2 = data['description'].apply(clean_text)
    text3 = data['content'].apply(clean_text)
    # title,description,content连接到一起
    text = text1 + text2 + text3
    text = text.apply(seg_text)
    db.close()
    return text,data




# 2、保存到数据库的一个新表当中，连同原有内容。
def insert_new(data,tablename,user,password,host,port,dbname):
    yconnect = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(user,password,host,port,dbname))
    pd.io.sql.to_sql(data,tablename, yconnect, schema=dbname, if_exists='append',index = False)

    

# 3、插入数据到旧的表中
def insert_raw(prediction,label,savedata,tabelname,user,password,host,port,dbname):
    db = link_sql(user,password,host,port,dbname)
    sqlmd = "UPDATE {} SET prediction = {},label = '{}' WHERE id = {}"
    cursor = db.cursor()
    try:
        cursor.execute(" alter table {} add column prediction int(10) ,add column label varchar(10);".format(tabelname))
    except:
        pass
    for i in range(len(label)):
        cursor.execute(sqlmd.format(tabelname,int(prediction[i]),label[i],int(savedata.iloc[i][0])))
    cursor.close()
    db.commit()
    db.close()