#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :Train_GPN.py
@说明        :用来辅助训练GPN网络
@时间        :2024/03/03 10:48:59
@作者        :leo
@版本        :1.0
'''
import torch
import numpy as np
from tqdm import tqdm
import Utils as util
import network.ModelOperation as mo
from torch.utils.data import DataLoader
import SystemModel as nm
import logging
from DataOperation import read_data,create_test_data
from train.EarlyStopping import EarlyStopping
logger = util.create_logging(file_name = "train_GPN")

#**********************************#
# 训练GPN模型
# 检查在验证集上的loss变化
# 检查在验证集上的吞吐量变化
#**********************************#


def train_GPN(param,data_meta,model,train_dataset,validate_dataset,output_folder):
    start_time = util.get_current_time_str()
    util.write_hyperparameter(output_folder, data_meta, param)
    loss_array = []
    validate_loss_array = []
    validate_throughput_array = []
    train_dataloader = DataLoader(train_dataset, batch_size=param["batch_size"], shuffle=True)
    early_stopping = EarlyStopping(patience = 50, path = output_folder.joinpath("es_dic.pt"))
    criterion = param["criterion"]
    optimizer = param["optimizer"]
    for epoch in tqdm(range(param["n_epoch"])):
        epoch_loss = 0
        count = 0
        for batch_index, data in enumerate(train_dataloader):
            X = data['Points'].cuda()
            Y = data['Solution'].cuda()
            o, p = model(X)
            # 深拷贝一个向量
            o = o.contiguous().view(-1, o.size()[-1])
            Y = Y.view(-1)
            loss = criterion(o, Y)
            epoch_loss += loss.item()
            count = count + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = epoch_loss / (count + 1)
        # logging.info("epoch loss:"+str(epoch_loss))
        v_loss = validate_loss(model,param,data_meta,validate_dataset)
        # logging.info("validation loss:"+str(v_loss))
        v_throughput,best_throughput,test_throughput = predict_throughput(data_meta,validate_dataset,model)
        logging.info("validation avg throughput:{}, best avg throughput {}, test avg throughput {}".format(str(v_throughput),str(best_throughput),str(test_throughput)) ) 
        loss_array.append(epoch_loss)
        validate_loss_array.append(v_loss)
        validate_throughput_array.append(v_throughput)
        # 是否要早停
        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping at {} epoch".format(epoch))
            torch.save(model, output_folder.joinpath('stop.pt'))
            break
        if epoch == (param["n_epoch"] - 1):
            save_root = output_folder.joinpath('final.pt') 
            torch.save(model, save_root)
        elif (epoch + 1) % param["save_frequence"] == 0:
            save_root = output_folder.joinpath(str(epoch + 1) + '.pt')
            torch.save(model, save_root)
    finish_time = util.get_current_time_str()
    util.write_train_log(output_folder, loss_array, validate_loss_array,validate_throughput_array)


def validate_loss(model,param,data_meta,validate_dataset):
    sequence_length = data_meta["sequence_length"]
    criterion = param["criterion"]
    validate_X,validate_Y = read_data(sequence_length,validate_dataset)
    validate_X = validate_X.cuda()
    validate_Y = validate_Y.cuda()
    validate_size = len(validate_dataset)
    validate_X = torch.zeros((validate_size, sequence_length, 2), dtype=torch.float32)
    validate_Y = torch.zeros((validate_size, sequence_length), dtype=torch.int64)
    for i in range(validate_size):
        validate_X[i] = validate_dataset[i]['Points']
        validate_Y[i] = validate_dataset[i]['Solution']
    validate_X = validate_X.cuda()
    validate_Y = validate_Y.cuda()
    val_predicted_probability = mo.infer_GPN_probability(model, validate_X)
    val_predicted_probability = val_predicted_probability.contiguous().view(-1, val_predicted_probability.size()[-1])
    validate_Y = validate_Y.view(-1)
    loss = criterion(val_predicted_probability, validate_Y)
    return loss.item()


def predict_throughput(data_meta,dataset,model):
    sequence_length = data_meta["sequence_length"]
    X,Y = read_data(sequence_length,dataset)

    net_output = mo.infer_GPN(model, X.cuda())
    net_output = net_output.cpu().numpy()
    # 验证集合结果
       
    print(nm.multi_calculate_throughput(Y[0:1],X[0:1]))
    print(nm.multi_calculate_throughput(net_output[0:1],X[0:1]))
    
    throughput_array = nm.multi_calculate_throughput(net_output,X)
    best_array = nm.multi_calculate_throughput(Y,X)
    # 测试集结果
    test_X = create_test_data(100,8,100)
    test_Y = mo.infer_PTN(model, test_X)
    test_array = nm.multi_calculate_throughput(test_Y,test_X)
    return np.mean(np.array(throughput_array)),np.mean(np.array(best_array)),np.mean(np.array(test_array))

    
