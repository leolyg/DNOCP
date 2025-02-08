# 文件主要用来描述网络的模型
# 目前只考虑SISO的情况
# 一个基站多个信道
# 每个信道由两个用户以NOMA方式占用
# 根据信道分配的方案计算出最终的
# version 1.0 (Last edited: 2021-12-30)
# author leo
import torch
import math
import numpy as np
from itertools import permutations
# 设置输出精度为小数点后6位
np.set_printoptions(precision=6)

##---------------------------------##
# 定义一些基础方法
##---------------------------------##
def db2pow(x):
    y = pow(10, (x / 10))
    return y


def pow2db(x):
    y = 10 * math.log10(x)
    return y


# Define the channel gain functions based on the 3GPP Urban Micro in
# "Further advancements for E-UTRA physical layer aspects (Release 9)."
# 3GPP TS 36.814, Mar. 2010. Note that x is measured in m and that the
# antenna gains are included later in the code
def passloss_LOS(fc, distance):
    return db2pow(-28 - 20 * math.log10(fc) - 22 * math.log10(distance));


def passloss_NLOS(fc, distance):
    return db2pow(-22.7 - 26 * math.log10(fc) - 36.7 * math.log10(distance))


# 计算功率因子alpha的值
def calculate_alpha(weak_channel_gain):
    alpha = (weak_channel_gain * P + sigma2 - pow(2, C_weak) * sigma2) / (pow(2, C_weak) * weak_channel_gain * P)
    if alpha > 0.5:
        return 0.5
        # return alpha
    if alpha < 0:
        return 0.01
    else:
        return alpha
##---------------------------------##
# 定义一些基础方法
##---------------------------------##


##---------------------------------##
# 定义一些常量
##---------------------------------##
# Carrier frequency (in GHz)
fc = 3
# Bandwidth
B = 1e7
# Noise figure (in dB)
noiseFiguredB = 10
# Compute the noise power in dBm
sigma2dBm = -174 + 10 * math.log10(B) + noiseFiguredB
sigma2 = db2pow(sigma2dBm)  # mW
# Define the antenna gains at the source and destination.
antennaGainS = db2pow(5)
antennaGainD = db2pow(0)
# Define the maximum power of a channel
P = 1000  # mW
# define the basic throughput constraint of weak channel
C_weak = 3



def calculate_OMA_throughput(channel_scheme, user_coordinates):
    """
    # 计算方案的整体吞吐量(OMA)
    # 输入信道分配方案、距离数组
    # 输出当前方案下最优吞吐量
    Args:
        channel_scheme (array): 信道分配
        user_coordinates (array): 用户位置

    Returns:
        float: 吞吐量
    """
    distance_array = transfer_to_distance_array(user_coordinates)
    ue_num = channel_scheme.shape[0]  # 用户的数量
    channel_num = int(ue_num)
    total = 0
    for i in range(channel_num):
        distance_i = distance_array[i]
        channel_gain_i = antennaGainS * antennaGainD * passloss_NLOS(fc, distance_i)
        SINR_i = P * channel_gain_i / sigma2
        throughput_i = 0.5 * math.log2(1 + SINR_i)  # 取一半的速率
        total += throughput_i
    return total


def multi_calculate_OMA_throughput(channel_scheme_array, user_coordinates_array):
    """
    计算多个样本的吞吐量(OMA)

    Args:
        channel_scheme_array (array): 信道分配数组
        user_coordinates_array (array): 用户位置数组

    Returns:
        float: 吞吐量
    """
    batch_size = channel_scheme_array.shape[0]
    throughput_array = np.zeros(batch_size)
    for i in range(batch_size):
        throughput_array[i] = calculate_OMA_throughput(channel_scheme_array[i], user_coordinates_array[i])
    return throughput_array

'''
输入分配方案和用户的位置
计算出总吞吐量
'''
def calculate_throughput(channel_scheme, user_coordinates):
    """_summary_

    Args:
        channel_scheme (array): 信道分配
        user_coordinates (array): 用户位置

    Returns:
        float: 吞吐量
    """
    distance_array = transfer_to_distance_array(user_coordinates)
    return calculate_channel_scheme_throughput(channel_scheme, distance_array)


def multi_calculate_throughput(channel_schemes, user_coordinates_array):
    """
    更推荐使用
    输入批量的信道分配向量和批量的用户位置向量
    获得批量的传输速率
    Args:
        channel_schemes (_type_): _description_
        user_coordinates_array (_type_): _description_

    Returns:
        _type_: _description_
    """
    if torch.is_tensor(channel_schemes):
        channel_schemes = channel_schemes.cpu().numpy()
    if torch.is_tensor(user_coordinates_array):
        user_coordinates_array = user_coordinates_array.cpu().numpy()
    batched_distance_array = multiple_transfer_to_distance_array(user_coordinates_array)
    return multiple_calculate_channel_scheme_throughput(channel_schemes, batched_distance_array)

def calculate_channel_scheme_throughput(channel_scheme, distance_array):
    """
    # 计算方案的整体吞吐量(NOMA)
    # 输入信道分配方案、距离数组
    # 输出当前方案下最优吞吐量
    Args:
        channel_scheme (_type_): _description_
        distance_array (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 排除掉channel_scheme中-1的情况
    ue_num = np.nonzero(channel_scheme + 1)[0].shape[0]
    channel_num = int(ue_num / 2)
    NOMA_array = channel_scheme[0:ue_num].reshape(channel_num, 2)
    total = 0
    for i in range(channel_num):
        UE_i, UE_j = NOMA_array[i][0], NOMA_array[i][1]
        distance_i, distance_j = distance_array[UE_i], distance_array[UE_j]
        channel_gain_i = antennaGainS * antennaGainD * passloss_NLOS(fc, distance_i)
        channel_gain_j = antennaGainS * antennaGainD * passloss_NLOS(fc, distance_j)
        if channel_gain_i > channel_gain_j:
            alpha = calculate_alpha(channel_gain_j)
            SINR_i = alpha * P * channel_gain_i / sigma2
            SINR_j = (1 - alpha) * P * channel_gain_j / (channel_gain_j * alpha * P + sigma2)
        else:
            alpha = calculate_alpha(channel_gain_i)
            SINR_i = (1 - alpha) * P * channel_gain_i / (channel_gain_i * alpha * P + sigma2)
            SINR_j = alpha * P * channel_gain_j / sigma2
        throughput_i = math.log2(1 + SINR_i)
        throughput_j = math.log2(1 + SINR_j)
        sum_throughput = throughput_i + throughput_j
        total += sum_throughput
    return total



def multiple_calculate_channel_scheme_throughput(channel_schemes, distance_arrays):
    """
    # 计算所有方案的整体吞吐量数组
    # 输入多个信道分配方案、多距离数组
    # 输出所有方案下最优吞吐量数组

    Args:
        channel_schemes (_type_): _description_
        distance_arrays (_type_): _description_

    Returns:
        _type_: _description_
    """
    batch_size = channel_schemes.shape[0]
    throughput_array = np.zeros(batch_size)
    for i in range(batch_size):
        throughput_array[i] = calculate_channel_scheme_throughput(channel_schemes[i], distance_arrays[i])
    return throughput_array



def transfer_to_distance_array(sample):
    """
    计算距离数组
    输入样本包含每个用户的位置
    输出欧式距离数组

    Args:
        sample (_type_): _description_

    Returns:
        _type_: _description_
    """
    sequence_length = sample.shape[0]
    distance_array = np.zeros(sequence_length)
    for i in range(sequence_length):
        location_x = sample[i][0]
        location_y = sample[i][1]
        distance_array[i] = pow(location_x * location_x + location_y * location_y, 0.5)
    return distance_array


# 计算多个位置的距离数组
# 输入许多用户的位置
# 输出多个位置的欧式距离数组
def multiple_transfer_to_distance_array(samples):
    """

    Args:
        samples (_type_): _description_

    Returns:
        _type_: _description_
    """
    batch_size = samples.shape[0]
    sequence_length = samples.shape[1]
    multi_distance_array = np.zeros((batch_size, sequence_length))
    for i in range(batch_size):
        multi_distance_array[i] = transfer_to_distance_array(samples[i])
    return multi_distance_array



def optimal_solution(points):
    """
    # 根据位置计算最优的信道分配方案
    # 输入每个用户的位置
    # 输出最优的方案
    Args:
        points (_type_): _description_

    Returns:
        _type_: _description_
    """
    sequence_length = points.shape[0]
    all_schemes = permutations(np.arange(sequence_length))
    max_scheme = None
    max_value = 0
    distance_array = transfer_to_distance_array(points)
    for s in all_schemes:
        scheme = np.array(s)
        value = calculate_channel_scheme_throughput(scheme, distance_array)
        if value > max_value:
            max_value = value
            max_scheme = scheme
    #对其进行标准化处理
    #TODO: 有bug
    return np.sort(max_scheme.reshape(int(sequence_length/2),2),axis=0).reshape(-1)


'''
获得所有可能方案的速率
'''
def all_solution(points):
    sequence_length = points.shape[0]
    all_schemes = permutations(np.arange(sequence_length))
    distance_array = transfer_to_distance_array(points)
    solution_array = []
    value_array = []
    for s in all_schemes:
        scheme = np.array(s)
        solution_array.append(scheme)
        value = calculate_channel_scheme_throughput(scheme, distance_array)
        value_array.append(value)
    return solution_array, value_array


# 只能用来验证，每次都要调用optimal_solution，非常耗时
def calculate_approximate_ratio(sample, solution):
    optimal = optimal_solution(sample)
    return calculate_approximate_ratio(sample, optimal, solution)


# 计算方案的优化近似比
# 输入每个用户的位置，最优方案，预测方案
# 输出近似比
def calculate_approximate_ratio(points, optimal, solution):
    distance_array = transfer_to_distance_array(points)
    opt_value = calculate_channel_scheme_throughput(optimal, distance_array)
    solution_value = calculate_channel_scheme_throughput(solution, distance_array)
    return solution_value / opt_value


# 计算所有分配方案的近似比
# 输入序列长度，通信范围
# 输出每种方案的近似比
def calculate_all_permutations_approximate_ratio(sequence_length, range):
    points = np.random.random((sequence_length, 2)) * range
    all_scheme = permutations(np.arange(sequence_length))
    distance_array = transfer_to_distance_array(points)
    throughput_array = []
    max = 0
    for scheme in all_scheme:
        scheme = np.array(scheme)
        value = calculate_channel_scheme_throughput(scheme, distance_array)
        if value > max:
            max = value
        throughput_array.append(value)
    return np.array(throughput_array) / max


if __name__ == '__main__':
    # print(calculate_all_permutations_approximate_ratio(6,50))
    # X = np.random.random((100,8,2))
    # print(X)
    # multi_distance_array = multiple_transfer_to_distance_array(X)
    # print(multi_distance_array)

    NOMA_total = calculate_channel_scheme_throughput(np.array([0, 1]), np.array([5, 50]))
    OMA_total = calculate_OMA_throughput(np.array([0, 1]), np.array([5, 50]))
    print(NOMA_total, OMA_total)

    for i in range(1, 300):
        for j in range(1, 300):
            NOMA_total = calculate_channel_scheme_throughput(np.array([0, 1]), np.array([i, j]))
            OMA_total = calculate_OMA_throughput(np.array([0, 1]), np.array([i, j]))
            if NOMA_total > OMA_total:
                print(i, j)
