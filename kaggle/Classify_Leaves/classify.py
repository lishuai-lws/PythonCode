# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'

from IPython import get_ipython


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision import models
import torch.nn as nn 
import shutil
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')



image = plt.imread('../data/classify-leaves/images/1.jpg')

plt.imshow(image)
print(image.shape)
plt.show()


data_dir = "..\\data\\classify-leaves"


batch_size = 128
valid_ratio = 0.1
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# # 读取数据集


train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, folder),
    transform = transform_train) for folder in ["train","train_valid"]]
valid_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, folder),
    transform = transform_test) for folder in ["valid","test"]]



train_dl, train_valid_dl = [torch.utils.data.DataLoader(dataset, batch_size,shuffle=True, drop_last=True) 
    for dataset in (train_ds, train_valid_ds)]
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size,shuffle=False, drop_last=False)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size,shuffle=False, drop_last=False)


# # 定义网络


net = models.resnet50(pretrained=True)
net_in_feature = net.fc.in_features
net.fc = nn.Linear(net_in_feature, 176)

# # 训练模型

def train(net,):
    pass




