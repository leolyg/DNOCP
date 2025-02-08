#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :SLPTN.py
@说明        :有监督学习PTN效果
@时间        :2024/03/04 08:26:52
@作者        :leo
@版本        :1.0
'''


import torch
from network.PointerNet import PointerNet
import Utils as util
import train.Train_PTN as tptn
from DataOperation import get_dataset_as_one
torch.cuda.empty_cache()
device = util.use_best_GPU()

#**********************************#
# 设置参数，有监督训练PTN模型
#**********************************#



# ================================
# 需要填入的参数，数据集的名称
# 示例：
# train_data_folder = "medium_data"
# validate_data_folder = "small_data"

train_data_folder = "2025-01-17-21-10-46"
validate_data_folder = "2025-01-17-21-13-22"
# ================================

train_data_path = util.get_data_folder().joinpath(train_data_folder)
validate_data_path = util.get_data_folder().joinpath(validate_data_folder)
data_size, sequence_length, distance_array = util.get_data_info(train_data_path)

# 生成的模型文件夹
dir_path = util.create_new_model_dir("slptn-",subfolder="SLPTN")

# 数据相关的信息
data_meta = {}
data_meta["DATA_FOLDER_PREFIX"] = train_data_path
data_meta["VALIDATE_DATA_FOLDER_PREFIX"] = validate_data_path
data_meta["data_size"] = data_size
data_meta["sequence_length"] = sequence_length
data_meta["distance_array"] = distance_array

# 神经网络参数相关
param = {
    "n_epoch": 10000,
    "batch_size": 512,
    "embedding_size": 128,
    "n_hidden": 64,
    "nof_lstms": 2,
    "dropout": 0,
    "bidir": True,
    "learning_rate": 0.0001,
    "save_frequence": 1000
}

# 获取数据集
train_dataset = get_dataset_as_one(data_meta['DATA_FOLDER_PREFIX'])
validate_dataset = get_dataset_as_one(data_meta['VALIDATE_DATA_FOLDER_PREFIX'])

model = PointerNet(param["embedding_size"],
                   param["n_hidden"],
                   param["nof_lstms"],
                   param["dropout"],
                   param["bidir"]).to(device)

criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=param["learning_rate"], momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=param["learning_rate"], betas=(0.9, 0.99))  # Adam
param['criterion'] = criterion
param['optimizer'] = optimizer

tptn.train_PTN(param,data_meta,model,train_dataset,validate_dataset,dir_path)

