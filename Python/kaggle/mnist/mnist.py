#!/usr/bin/env python
# coding: utf-8

# In[21]:


import torch
import numpy as np
import os
import pandas as pd
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt 
from torchvision import transforms,models


# In[22]:


SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed()


# In[23]:


# 训练数据路径
train_df = pd.read_csv("../data/mnist/train.csv",dtype=np.float32)
# print(train_df.dtypes)
# 取出数据中的标签
labels = train_df.pop('label').astype('int64')
# print(labels.dtype)


# In[24]:


train_df = train_df/255.0
# print(train_df.dtypes)
# print(type(train_df))
# 转为numpy.ndarray类型，notebook运行时要全部运行，不然会出现已经是numpy.ndarray的错误
train_df = train_df.to_numpy()
labels = labels.to_numpy()

# 输入转为28*28*1，输出转为1维
train_df = train_df.reshape(-1,28,28,1)
labels = labels.reshape(-1,1)
# print(type(train_df))
# print(labels)


# In[25]:


# 划分训练集和测试集
x_train,x_val,y_train,y_val = train_test_split(train_df,labels,test_size=0.2,random_state=SEED)
# print('y_train:',y_train)
# print('y_val:',y_val)


# In[26]:


# 创建数据集的类
class MNISTDataset(Dataset):
    def __init__(self,images,labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self,index):
        label = self.labels[index]
        image = self.images[index]
        image = self.transform(image)
        
        image = image.repeat(3,1,1)
        # image.size(3,28,28)        
        return image,label
    def __len__(self):
        return len(self.images)


# In[27]:


# 创建数据集
train_data = MNISTDataset(x_train,y_train)
val_data = MNISTDataset(x_val,y_val)


# In[28]:


# 设置batch_size
BATCH_SIZE = 64


# In[29]:


# 加载数据集
train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_data,batch_size=BATCH_SIZE)


# In[30]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[31]:


NUM_CLASSES=10
EPOCHS=30


# In[32]:


# 加载resnet模型，并进行预训练
model = models.resnet18(pretrained=True)
# 获取fc层中固定的参数
num_ftrs = model.fc.in_features
# 修改为分类类别
model.fc = nn.Linear(num_ftrs,NUM_CLASSES)
model.to(device)


# In[33]:


# 设置损失函数为交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器 Adam是一种自适应学习率的优化方法，Adam利用梯度的一阶矩估计和二阶矩估计动态的调整学习率。
# amsgrad- 是否采用AMSGrad优化方法，asmgrad优化方法是针对Adam的改进，通过添加额外的约束，使学习率始终为正值。
optimizer = torch.optim.Adam(model.parameters(),amsgrad=True)
# Decays the learning rate of each parameter group by gamma every step_size epochs. 
xp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1,verbose=True)


# In[ ]:


# 训练模型
for epoch in range(EPOCHS):
    for i,(images,labels) in enumerate(train_loader):
        # 可以使用gpu
        images = images.to(device)
        labels = labels.to(device)
        # Forward
        # 梯度清零
        optimizer.zero_grad()
        # 输入模型计算预测值
        pred = model(images)
        # 计算损失
        loss = criterion(pred,labels.flatten())

        # backward and optimize 
        # 损失传递
        loss.backward()
        # 更新
        optimizer.step()

        if (i+1) % 300 == 0:
            print(f'Epoch:{epoch+1}/{EPOCHS},Loss:{loss.item()}')
    
    xp_lr_scheduler.step()
# 测试时，不启用 Batch Normalization 和 Dropout。调用model.eval()
mode.eval()
with torch.no_grad():
    currect =0
    total = 0
    for images,labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        按行取最大值，predicted里保存index
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        currect += (predicted == labels.flatten()).sum()
    
    print(f'Validation acc :{100 * currect/total}')


# In[ ]:


# Submit result
# test_df = pd.read_csv('../data/mnist/test.csv',dtype=np.float32)
# test_df = test_df.to_numpu()/255.0
# test_df = test_df.reshape(-1,28,28,1)
# test_tensor = torch.from_numpy(test_df).permute(0, 3, 1, 2)
# test_tensor = test_tensor.repeat(1, 3, 1, 1)
# images= test_tensor.to(device)
# outputs = model(images)
# _, predictions = torch.max(outputs, 1)
# predictions = predictions.cpu()
# submission = pd.DataFrame({'ImageId': np.arange(1, (predictions.size(0) + 1)), 'Label': predictions})
# submission.to_csv("submission.csv", index = False)

