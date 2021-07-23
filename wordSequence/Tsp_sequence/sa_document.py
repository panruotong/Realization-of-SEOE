import numpy as np
from sko.SA import SA_TSP
from wordSequence.Tsp_sequence import functions
import time

# 读入样本、属性
sample = np.load('../data/q2_sample_line.npy')
feature = np.load('../data/q2_feature_line.npy')
# sample = np.load('../data/sample_line.npy')
# feature = np.load('../data/feature_line.npy')

# 读入feature.name文件
f_name = [];
with open('../data/q2_feature_name.txt', 'r') as f_name_file:
    for line in f_name_file:
        f_name.append(line.strip())

# path
path_document = '../data/sa/q2_sa_document.seq'
path_time = '../data/sa/q2_sa_time.txt'
# path_document = '../data/sa_document.seq'
# path_time = '../data/sa_time.txt'


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# 循环处理每一个样本
total_time = 0
average_time = 0
total_path = 0
for e in range(sample.shape[0]):
    nums = sample[e]
    feature_name, distance_matrix = functions.feature(nums, f_name, feature)
    num_points = distance_matrix.shape[0]
    # 求最短路径
    start = time.perf_counter()
    # x0：初始解  T_max：初始温度  T_min：终止温度  L：在每个温度下的迭代次数
    sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=100, T_min=1, L=10 * num_points)
    best_points, best_distance = sa_tsp.run()
    end = time.perf_counter()
    total_time = total_time + (end - start)
    total_path = total_path + best_distance
    # 写入document
    sample_document = []
    for m in range(len(best_points)):
        j = best_points[m]
        sample_document.append(feature_name[j])
    functions.document(path_document, sample_document)
    time_line = str(e) + "：\t" + str(end - start) + 's\t' + str(best_distance) + '\n'
    functions.run_time(path_time, time_line)

average_time = total_time / sample.shape[0]
average_path = total_path / sample.shape[0]
average_time_line = '平均运行时间：' + str(average_time) + 's\n'
average_path_line = '平均路径长度：' + str(average_path)
functions.run_time(path_time, average_time_line)
functions.run_time(path_time, average_path_line)