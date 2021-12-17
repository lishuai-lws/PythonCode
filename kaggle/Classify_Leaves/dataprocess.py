import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
import shutil
import matplotlib.pyplot as plt
import collections 
import math

#数据文件夹路径
data_dir = "..\\data\\classify-leaves"
out_dir = "..\\data\\classify-leaves"


def read_csv_labels(fname):
    df = pd.read_csv(fname)
    return np.array(df)

# 将filename复制到文件夹target_dir下
def copyfile(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def  reorg_train_valid(data_dir,out_dir, train_labels, valid_ratio):
    #统计训练集类别中数量最少的个数。
    n = collections.Counter(train_labels[:,1]).most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n*valid_ratio))
    label_count = {}
    for label in train_labels:
        fname = os.path.join(data_dir, label[0])
        # copyfile(fname, os.path.join(out_dir,'train_valid',label[1]))
        if label[1] not in label_count or label_count[label[1]] < n_valid_per_label:
            copyfile(fname, os.path.join(out_dir,'valid',label[1]))
            label_count[label[1]] = label_count.get(label[1],0) + 1 
        else:
            copyfile(fname, os.path.join(out_dir,'train',label[1]))
        print("整理图片名称:{},类别：{},".format(label[0],label[1]))
    print("训练集,测试集整理完成！")
train_labels = read_csv_labels(os.path.join(data_dir,'train.csv'))



test_images = np.array(pd.read_csv(os.path.join(data_dir,'test.csv')))
def reorg_test(data_dir):
    test_images = np.array(pd.read_csv(os.path.join(data_dir,'test.csv')))
    for image in test_images:
        fname = os.path.join(data_dir,image[0])
        copyfile(fname, os.path.join(out_dir,'test',"unknown"))
    print('整理的测试集数据量为：',test_images.shape)


valid_ratio = 0.1
reorg_train_valid(data_dir, out_dir, train_labels, valid_ratio)
reorg_test(data_dir)
