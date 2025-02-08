#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :SLGPN.py
@说明        :监督学习训练GPN的核心
@时间        :2024/03/03 16:46:47
@作者        :leo
@版本        :1.0
'''
import torch
import torch.nn as nn
import Utils as util
import train.Train_GPN as tgpn
from network.GraphPointNet import GPN
from DataOperation import get_dataset_as_one

torch.cuda.empty_cache()
device = util.use_best_GPU()

#**********************************#
# 设置参数，有监督训练GPN模型
#**********************************#


# ================================
# 需要填入的参数，数据集的名称
# 数据量为10000，用户数量为8，多距离混合数据
# 目前使用小数据集
# 如果使用其它数据集：
# 中数据集
# DATA_FOLDER_PREFIX = basic_path+"/data/medium_data/"
# 大数据集
# DATA_FOLDER_PREFIX = basic_path+"/data/large_data/"
# 示例：
# train_data_folder = "large_data"
# validate_data_folder = "small_data"

train_data_folder = "2025-01-17-21-10-46"
validate_data_folder = "2025-01-17-21-13-22"
# ================================

train_data_path = util.get_data_folder().joinpath(train_data_folder)
validate_data_path = util.get_data_folder().joinpath(validate_data_folder)
data_size, sequence_length, distance_array = util.get_data_info(train_data_path)

# 生成的模型文件夹
dir_path = util.create_new_model_dir("slgpn-",subfolder="SLGPN")

# 数据相关的信息
data_meta = {}
data_meta["DATA_FOLDER_PREFIX"] = train_data_path
data_meta["VALIDATE_DATA_FOLDER_PREFIX"] = validate_data_path
data_meta["data_size"] = data_size
data_meta["sequence_length"] = sequence_length
data_meta["distance_array"] = distance_array

# 神经网络参数相关
param = {
    "n_epoch": 3000,  # epochs
    "learning_rate": 0.0001,  # learning rate
    "batch_size": 512,  # batch_size
    "n_hidden": 256,
    "save_frequence": 200  # save model frequence
}

train_dataset = get_dataset_as_one(data_meta['DATA_FOLDER_PREFIX'])
validate_dataset = get_dataset_as_one(data_meta['VALIDATE_DATA_FOLDER_PREFIX'])
model = GPN(n_feature=2, n_hidden=param["n_hidden"]).to(device)

'''
# optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate) # SGD
# optimizer = torch.optim.SGD(model.parameters(), lr=param["learn_rate"], momentum=0.9)  # Momentum
# optimizer = torch.optim.RMSprop(model.parameters(),lr=param["learn_rate"],alpha=0.9)  # RMSprop
# optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate,betas=(0.9,0.99)) # Adam
# optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate) # Adam
'''

optimizer = torch.optim.Adam(model.parameters(), lr=param["learning_rate"], betas=(0.9, 0.99))  # Adam
criterion = nn.CrossEntropyLoss()
param['criterion'] = criterion
param['optimizer'] = optimizer


tgpn.train_GPN(param,data_meta,model,train_dataset,validate_dataset,dir_path)
