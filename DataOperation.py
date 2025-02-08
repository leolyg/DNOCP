#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :DataGenerator.py
@说明        :用来生成数据
@时间        :2024/03/02 19:35:54
@作者        :leo,winkemoji
@版本        :1.0
'''

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import SystemModel as nm
from multiprocessing import Process, Manager, Queue, Lock
import time
import traceback
import datetime
import os
from glob import glob
import json
import torch.utils.data as data_util


def preprocess(label):
    label = label[np.argsort(label[:,0])]
    return label
    

def create_test_data(num,sequence_length,distance):
    return torch.from_numpy(np.random.random((num, sequence_length, 2)) * distance).cuda().to(torch.float32)
    
    
    


def read_data(sequence_length,dataset):
    """
    读取之前生成的数据

    Args:
        sequence_length (int): 序列的长度
        dataset (Dataset): 数据集（可以是合并后的，也可以是单独的子数据集）

    Returns:
        Tensor: 读取数据为Tensor
    """

    size = len(dataset)
    X = torch.zeros((size, sequence_length, 2), dtype=torch.float32)
    Y = torch.zeros((size, sequence_length), dtype=torch.int64)
    for i in range(size):
        X[i] = dataset[i]['Points']
        Y[i] = dataset[i]['Solution']
    return X,Y

def get_multi_dataset(data_path):
    """
    从数据文件夹中获取多个数据集

    Args:
        data_path (str): 数据文件夹路径

    Returns:
        _type_: _description_
    """
    paths = os.listdir(data_path)
    multi_datasets= []
    for i in range(len(paths)):
        subset_path = data_path.joinpath(paths[i])
        dataset = DNODataset(dataset_folder_path=subset_path)
        multi_datasets.append(dataset)
    return multi_datasets

def get_dataset_as_one(data_path):
    """
    将从数据文件夹中获取的多个数据集进行合并
    Args:
        data_path (str): 数据文件夹路径

    Returns:
        _type_: _description_
    """
    paths = os.listdir(data_path)
    multi_datasets= []
    for i in range(len(paths)):
        subset_path = data_path.joinpath(paths[i])
        dataset = DNODataset(dataset_folder_path=subset_path)
        multi_datasets.append(dataset)
        single_dataset = data_util.ConcatDataset(multi_datasets)
    return single_dataset


class DNODataset(Dataset):
    """用于生成用户数据集
    """

    def __init__(self, **kwargs):
        """初始化，传入参数, kwargs为字典
        需要键包括{dataset_folder_path} 或者 {data_size, sequence_length, distance_range, num_workers}
        """
        if 'dataset_folder_path' in kwargs:
            self.load_data(kwargs['dataset_folder_path'])
            return

        self.folder_path = kwargs['folder_path']
        self.data_size = kwargs['data_size']
        self.sequence_length = kwargs['sequence_length']
        self.distance_range = 200 if not 'distance_range' in kwargs else kwargs[
            'distance_range']
        self.num_workers = kwargs['num_workers']
        self.data_pool = self.async_generate_data()
        self.save_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data_pool['Points_List'][idx]).float()
        solution = torch.from_numpy(self.data_pool['Solutions'][idx]).long()
        sample = {'Points': tensor, 'Solution': solution}
        return sample

    def __datasize_per_worker(self):
        """计算每个进程(worker)处理计算多少个数据

        Raises:
            Exception: 要求每一个进程处理数据量相同，如果总数据量无法平均分配给每个进程，引起异常

        Returns:
            int: 每个进程处理数据量
        """
        if self.data_size % self.num_workers != 0:
            raise Exception(
                "[error]data_size divide num_workers must be integer.")
        return int(self.data_size / self.num_workers)

    def async_generate_data(self):
        """异步生成数据

        Raises:
            Exception: 如果最后worker生成的数据量的总和与预计生成数量不匹配，抛出丢失数据异常

        Returns:
            dict: 保存生成数据的数据池
        """
        meta = self.meta()
        print("ready to generate data.")
        print("----------------------")
        print('data_size      : %s' % meta['data_size'])
        print('sequence_length: %s' % meta['sequence_length'])
        print('distance       : %s' % meta['distance'])
        print("----------------------")
        time_start = time.time()
        sync_lock = Lock()
        data_pool = Manager().dict()
        data_pool['Points_List'] = []
        data_pool['Solutions'] = []

        q = Queue()
        workers = [Process(target=self.generate_data, args=(worker_id, q, data_pool, sync_lock,))
                   for worker_id in range(self.num_workers)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        time_end = time.time()
        print('time cost %.2fs' % (time_end - time_start))
        if len(data_pool['Points_List']) == len(
                data_pool["Solutions"]) == self.data_size:
            print('generated %s data.' % self.data_size)
        else:
            raise Exception('[error]missing generated data....')
        return data_pool

    def save_data(self):
        """保存生成的数据, 默认放在./data目录下
        """
        folder_path = self.folder_path + '/%s-%s-%s'%(self.data_size, self.sequence_length, self.distance_range)
        #随机生成的
        points_list = self.data_pool['Points_List']
        solutions = self.data_pool['Solutions']
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(points_list, folder_path + '/points.pt')
        torch.save(solutions, folder_path + '/solutions.pt')
        with open(folder_path + '/meta.json', "w") as f:
            f.write(json.dumps(self.meta()))
        print("data stored into %s" % folder_path)

    def load_data(self, folder_path):
        """读取数据

        Args:
            folder_path (string) 数据保存的文件夹路径

        Raises:
            FileExistsError: 如果在文件夹下没有找到对应文件, 抛出异常
        """
        if not os.path.exists(folder_path):
            raise FileExistsError("folder not found.")
        folder_path = str(folder_path)
        points_list_pattern = folder_path + '/points.pt'
        solutions_pattern = folder_path + '/solutions.pt'
        points_list_path = glob(points_list_pattern)[0]
        solutions_path = glob(solutions_pattern)[0]
        with open(folder_path + '/meta.json', 'r') as f:
            meta = json.load(f)
        points_list = torch.load(points_list_path)
        solutions = torch.load(solutions_path)

        data_pool = {
            'Points_List': points_list,
            'Solutions': solutions
        }
        self.data_pool = data_pool
        self.data_size = meta['data_size']
        self.sequence_length = meta['sequence_length']
        self.distance_range = meta['distance']

    def generate_data(self, worker_id, q, data_pool, lock):
        """同步生成数据(此函数不会暴露在外部使用)

        Args:
            worker_id (int): 用来描述当前进程，同时也用于设置随机种子
            q (queue): 消息队列，原用来通知处理完毕，现暂时没在使用
            data_pool (dict): 加了同步锁的数据池
            lock (lock): 同步锁
        """
        try:
            points_list = []
            solutions = []
            data_size = self.__datasize_per_worker()
            np.random.seed(worker_id)  # 设置随机种子, 防止生成的数据一样
            data_iter = tqdm(range(data_size), unit='data')
            for i, _ in enumerate(data_iter):
                data_iter.set_description(
                    'Worker%s Data points %i/%i' % (worker_id, i + 1, data_size))
                points_list.append(np.random.random(
                    (self.sequence_length, 2)) * self.distance_range)
            solutions_iter = tqdm(points_list, unit='solve')
            for i, points in enumerate(solutions_iter):
                solutions_iter.set_description(
                    'Worker%s Solved %i/%i' % (worker_id, i + 1, len(points_list)))
                solutions.append(nm.optimal_solution(points))
            lock.acquire()
            data_pool['Points_List'] += [] + points_list
            data_pool['Solutions'] += [] + solutions
            lock.release()
            q.put({'worker_status': True})
        except BaseException:
            traceback.print_exc()
        finally:
            q.put({'worker_status': False})

    def meta(self):
        """返回当前数据集的元数据

        Returns:
            dict: 元数据，包括数据集长度，每项点的个数，点距离基站的距离
        """
        return {
            'data_size': self.data_size,
            'sequence_length': self.sequence_length,
            'distance': self.distance_range
        }


if __name__ == '__main__':
    # 根据当前的时间生成一个文件夹，存放所有的数据
    folder_path = './data/' + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H-%M-%S")  # e.g. ./data/2022-01-11-06-50-18
    for i in range(7):
        dataset = DNODataset(
            folder_path=folder_path,
            data_size=100, sequence_length=8, distance_range=100+i*50, num_workers=5)
