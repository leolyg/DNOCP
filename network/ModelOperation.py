#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :operation.py
@说明        :模型的一些辅助操作
@时间        :2024/03/03 16:03:12
@作者        :leo
@版本        :1.0
'''
#**********************************#
# 模型主要的一些辅助操作
# 主要用于获取模型的输出
# 有些情况需要输出概率
# 有些情况需要直接输出方案
#**********************************#

import SystemModel as nm
import numpy as np
import torch

def infer_GPN_probability(model,X):
  """

  Args:
      model (GPN): 模型
      X (Tensor): 位置信息

  Returns:
      torch: 各个pointer概率
  """
  output,pointer = model(X)
  return output
  # batch_size = X.size()[0]
  # size = X.size()[1]
  # mask = torch.zeros(batch_size, size).cuda()
  # x = torch.zeros([batch_size, 2], dtype=torch.float).cuda()
  # predicted_probability = torch.zeros([batch_size, size, size]).cuda()
  # h = None
  # c = None
  # ###预测的每一步
  # for k in range(size):
  #   output, h, c, _ = model(x=x, X_all=X, h=h, c=c, mask=mask)
  #   idx = torch.argmax(output, dim=1)
  #   predicted_probability[:, k] = output.clone()
  #   x = X[[i for i in range(batch_size)], idx.data].clone()
  #   mask[[i for i in range(batch_size)], idx.data] += -np.inf
  # return predicted_probability

def infer_GPN(model,X):
    """

    Args:
        model (GPN): 模型
        X (Tensor): 位置信息

    Returns:
        torch: 输出的
    """
    output,pointer = model(X)
    return pointer

def infer_PTN_probability(model,X):
  output, pointer = model(X)
  return output


def infer_PTN(model,X):
  output, pointer = model(X)
  return pointer


# def gnn_model_throughput_array(model,test_data,multiple_distances=None):
#   net_output = infer_GPN(model,torch.from_numpy(test_data).to(torch.float32).cuda())
#   if multiple_distances is None:
#     multiple_distances = nm.multiple_transfer_to_distance_array(test_data)
#   net_output =net_output.cpu().numpy()
#   throughput_array = nm.multiple_calculate_channel_scheme_throughput(net_output,multiple_distances)
#   return np.array(throughput_array)

# def gnn_model_throughput(model,test_data,multiple_distances=None):
#   model_avg = np.mean(gnn_model_throughput_array(model,test_data,multiple_distances))
#   return model_avg


# def ptn_model_throughput_array(model,test_data,multiple_distances=None):
#   net_output = infer_PTN(model, torch.from_numpy(test_data).to(torch.float32).cuda())
#   if multiple_distances is None:
#     multiple_distances = nm.multiple_transfer_to_distance_array(test_data)
#   throughput_array = nm.multiple_calculate_channel_scheme_throughput(net_output, multiple_distances)
#   return np.array(throughput_array)

# def ptn_model_throughput(model,test_data,multiple_distances=None):
#   model_avg = np.mean(ptn_model_throughput_array(model,test_data,multiple_distances))
#   return model_avg
