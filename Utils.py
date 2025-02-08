#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :Utils.py
@说明        :一些辅助函数
@时间        :2024/03/02 20:55:22
@作者        :leo,winkemoji
@版本        :1.0
'''

import datetime
import os
import pathlib
import numpy as np
import pytz
from datetime import datetime
import time
from DataOperation import DNODataset
import torch.utils.data as data_util
from pynvml import *
import torch
import logging
from logging.handlers import RotatingFileHandler

def get_project_folder():
    return os.path.dirname(os.path.realpath(__file__))

def get_base():
    path = pathlib.Path(__file__).parent.resolve()
    return path

def get_data_folder():
    return get_base().joinpath("data")

def get_log_path():
    return get_base().joinpath("logs")    

def get_model_path():
    return get_base().joinpath("model")

def create_logging(file_name):
    # 创建日志记录器
    logger = logging.getLogger()
    # 设置日志级别
    logger.setLevel(logging.INFO)
    # 创建日志记录器，指明日志保存的路径、每个日志文件的最大大小、保存的日志文件个数上限
    file_log_handler = RotatingFileHandler(get_log_path().joinpath(file_name+".logs"), maxBytes=1024 * 1024 * 100, backupCount=10)
    # 创建日志记录的格式
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s')
    # 设置日志记录格式
    file_log_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()  # 往屏幕上输出
    # 添加日志记录器
    logger.addHandler(file_log_handler)
    #logger.addHandler(stream_handler)
    # 不输出到控制台
    logger.propagate = False
    return logger

def get_dataset(DATA_FOLDER_PREFIX):
    """
    获取数据集
    由于训练时将多个集合合并，因此需要手动合并一下
    Args:
        DATA_FOLDER_PREFIX (str): _description_

    Returns:
        _type_: _description_
    """
    paths = os.listdir(DATA_FOLDER_PREFIX)
    multi_datasets= []
    for i in range(len(paths)):
        data_path = DATA_FOLDER_PREFIX+paths[i]
        dataset = DNODataset(dataset_folder_path=data_path)
        multi_datasets.append(dataset)
        single_dataset = data_util.ConcatDataset(multi_datasets)
    return single_dataset


def create_new_model_dir(prefix,subfolder=None):
    current_time = get_current_time_str()
    model_path = get_model_path()
    if subfolder != None:
        model_path = model_path.joinpath(subfolder)
        if os.path.exists(model_path) != True:
            os.mkdir(model_path)
    dir_path = model_path.joinpath(prefix + current_time)
    if os.path.exists(dir_path) != True:
        os.mkdir(dir_path)
    return dir_path

# 模型输出文件夹前缀
OUTPUT_PATH_PREFIX = "./output/"
'''
获取输出的文件夹名
'''
def init_output(data):
    """训练前初始化输出

    Args:
        data (dict): 模型的元数据

    Returns:
        string: 模型保存的文件夹路径
    """
    sub_folder_name = datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    # e.g. ./output/2022-01-04 19.43.56
    folder_path = OUTPUT_PATH_PREFIX + sub_folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    meta_path = folder_path + "/meta.txt"
    with open(meta_path, 'w') as f:
        for k, v in data["model_meta"].items():
            f.write("%s=%s\n" % (k, v))
        for k, v in data["dataset_meta"].items():
            f.write("%s=%s\n" % (k, v))
        for k, v in data["train_meta"].items():
            f.write("%s=%s\n" % (k, v))
        f.write("use_cuda=%s\n" % (data["USE_CUDA"]))
    return folder_path



'''
获得时间戳的字符串
'''
def get_current_time_str():
  # 北京时间
  utc = pytz.utc
  beijing = pytz.timezone("Asia/Shanghai")
  # 时间戳
  loc_timestamp = time.time()
  # 转utc时间 datetime.datetime 类型
  utc_date = datetime.utcfromtimestamp(loc_timestamp)
  # 转utc当地 标识的时间
  utc_loc_time = utc.localize(utc_date)
  fmt = '%Y-%m-%d-%H-%M-%S'
  # 转北京时间
  beijing_time = utc_loc_time.astimezone(beijing)
  cst_time = beijing_time.strftime(fmt)
  return cst_time



def get_data_info(path):
    dirs = os.listdir(path)
    distance_array = []
    data_size = 0
    sequence_length = 0
    for file in dirs:
        file_meta = file.split('-')
        distance_array.append(int(file_meta[2]))
        data_size = int(file_meta[0])
        sequence_length = int(file_meta[1])
    return data_size,sequence_length,distance_array


def write_hyperparameter(dir_path,data,param):
    dir_path = str(dir_path)
    f = open(dir_path + '/meta.txt', 'w')
    for k,v in data.items():
        f.write(str(k)+':' + str(v) + '\n')
    for k,v in param.items():
        f.write(str(k)+':' + str(v) + '\n')

def write_log(dir_path,loss_array,validate_loss_array):
    dir_path = str(dir_path)
    f = open(dir_path + '/loss_logs.txt', 'w')
    for i in range(len(loss_array)):
        f.write(str(loss_array[i]) + '\n')
    f.close()
    f = open(dir_path + '/val_log.txt', 'w')
    for i in range(len(validate_loss_array)):
        f.write(str(validate_loss_array[i]) + '\n')
    f.close()
    
def write_train_log(dir_path,loss_array,validate_loss_array,validate_throughput_array):
    f = open(dir_path.joinpath('loss_log.txt'), 'w')
    for i in range(len(loss_array)):
        f.write(str(loss_array[i]) + '\n')
    f.close()
    f = open(dir_path.joinpath('val_log.txt'), 'w')
    for i in range(len(validate_loss_array)):
        f.write(str(validate_loss_array[i]) + '\n')
    f.close()
    f = open(dir_path.joinpath('val_throughput_log.txt'), 'w')
    for i in range(len(validate_throughput_array)):
        f.write(str(validate_throughput_array[i]) + '\n')
    f.close()

'''
根据显卡的实际情况选择GPU
'''
def show_gpu():
    # 初始化
    nvmlInit()
    # 获取GPU个数
    deviceCount = nvmlDeviceGetCount()
    total_memory = 0
    total_free = 0
    total_used = 0
    gpu_name = ""
    gpu_num = deviceCount
    used_array = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle)#.decode('utf-8')
        # 查看型号、显存、温度、电源
        print("[ GPU{}: {}".format(i, gpu_name), end="    ")
        print("总共显存: {}G".format((info.total//1048576)/1024), end="    ")
        print("空余显存: {}G".format((info.free//1048576)/1024), end="    ")
        print("已用显存: {}G".format((info.used//1048576)/1024), end="    ")
        print("显存占用率: {}%".format(info.used/info.total), end="    ")
        print("运行温度: {}摄氏度 ]".format(nvmlDeviceGetTemperature(handle,0)))
        used_array.append(info.used/info.total)
        total_memory += (info.total//1048576)/1024
        total_free += (info.free//1048576)/1024
        total_used += (info.used//1048576)/1024
    print("显卡名称：[{}]，显卡数量：[{}]，总共显存；[{}G]，空余显存：[{}G]，已用显存：[{}G]，显存占用率：[{}%]。".format(gpu_name, gpu_num, total_memory, total_free, total_used, (total_used/total_memory)))
    #关闭管理工具
    nvmlShutdown()
    return used_array


def use_gpu(used_percentage=0.75):
    '''
    不使用显存占用率高于used_percentage的gpu
    :param used_percentage:
    :return:
    '''
    nvmlInit()
    gpu_num = nvmlDeviceGetCount()
    out = ""
    gpu_array = []
    for i in range(gpu_num):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_percentage_real = info.used / info.total
        if out == "":
            if used_percentage_real < used_percentage:
                out += str(i)
                gpu_array.append(i)
        else:
            if used_percentage_real < used_percentage:
                out += "," + str(i)
    nvmlShutdown()
    print("当前可使用GPU：",out)
    return out

def use_best_GPU():
    """
    使用最闲的GPU
    Returns:
        int: 最佳device id
    """
    used_array = show_gpu()
    best_index = np.argmin(np.array(used_array))
    print("当前闲置显存最大的GPU",best_index)
    return int(best_index)

def try_multi_GPU(used_percentage=0.75):
    used_array = show_gpu()
    nvmlInit()
    gpu_num = nvmlDeviceGetCount()
    out = ""
    gpu_array = []
    for i in range(gpu_num):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_percentage_real = info.used / info.total
        if out == "":
            if used_percentage_real < used_percentage:
                out += str(i)
                gpu_array.append(i)
        else:
            if used_percentage_real < used_percentage:
                out += "," + str(i)
                gpu_array.append(i)
    nvmlShutdown()
    num = len(gpu_array)
    if num > 1:
        print("当前使用多块GPU",out)
    else:
        print("当前可使用GPU：", out)
    return out,gpu_array


if __name__ == '__main__':
    show_gpu()
    #use_gpu(0.75)
    print(get_base())
