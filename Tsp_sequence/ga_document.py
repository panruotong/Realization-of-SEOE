import numpy as np
from sko.GA import GA_TSP
from wordSequence.Tsp_sequence import functions
import time

# 读入样本、属性
sample = np.load('../data/q2_sample_line.npy')
feature = np.load('../data/q2_feature_line.npy')
print(feature)
# sample = np.load('../data/sample_line.npy')
# feature = np.load('../data/feature_line.npy')

# 读入feature.name文件
f_name = [];
with open('../data/q2_feature_name.txt', 'r') as f_name_file:
    for line in f_name_file:
        f_name.append(line.strip())

print(len(f_name))
print(f_name)

# path
path_document = '../data/ga/q2_ga_document.seq'
path_time = '../data/ga/q2_ga_time.txt'
# path_document = '../data/ga_document.seq'
# path_time = '../data/ga_time.txt'


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
print(sample.shape[0])
for e in range(sample.shape[0]):
    nums = sample[e]
    # feature_name:样本中为1的选项名
    # distance_matrix:选项的海明距离矩阵
    feature_name, distance_matrix = functions.feature(nums, f_name, feature)
    # 求最短路径
    start = time.perf_counter()
    # size_pop：种群大小  max_iter：终止进化代数  prob_mut：变异概率
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=distance_matrix.shape[0], size_pop=50, max_iter=500, prob_mut=0.2)
    best_points, best_distance = ga_tsp.run()
    end = time.perf_counter()
    total_time = total_time + (end - start)
    total_path = total_path + best_distance
    # 写入document
    sample_document = []
    for m in range(len(best_points)):
        j = best_points[m]
        sample_document.append(feature_name[j])
    # print(sample_document)
    functions.document(path_document, sample_document)
    time_line = str(e) + "：\t" + str(end - start) + 's\t' + str(best_distance[0]) + '\n'
    functions.run_time(path_time, time_line)

average_time = total_time / sample.shape[0]
average_path = total_path / sample.shape[0]
print("运行到这里了")
average_time_line = '平均运行时间是：' + str(average_time) + 's\n'
average_path_line = '平均路径长度：' + str(average_path[0])
functions.run_time(path_time, average_time_line)
functions.run_time(path_time, average_path_line)

