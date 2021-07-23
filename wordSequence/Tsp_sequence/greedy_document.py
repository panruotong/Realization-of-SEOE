import numpy as np
from wordSequence.Tsp_sequence import common_function
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

print(len(f_name))
print(f_name)

# path
path_document = '../data/greedy/q2_greedy_document.seq'
path_time = '../data/greedy/q2_greedy_time.txt'

# 循环处理每一个样本
total_time = 0
average_time = 0
total_path = 0
for e in range(sample.shape[0]):
    nums = sample[e]
    feature_name, distance_matrix = common_function.feature(nums, f_name, feature)
    # 求最短路径
    i = 1
    n = distance_matrix.shape[0]
    j = 0
    sum_path = 0
    s = []
    s.append(0)
    start = time.perf_counter()

    while True:
        k = 1
        Detemp = 10000000
        while True:
            l = 0
            flag = 0
            if k in s:
                flag = 1
            if (flag == 0) and (distance_matrix[k][s[i - 1]] < Detemp):
                j = k;
                Detemp = distance_matrix[k][s[i - 1]];
            k += 1
            if k >= n:
                break;
        s.append(j)
        i += 1;
        sum_path += Detemp
        if i >= n:
            break;
    sum_path += distance_matrix[0][j]  # 回路
    end = time.perf_counter()

    total_time = total_time + (end - start)
    total_path = total_path + sum_path
    # 写入document
    sample_document = []
    for m in range(len(s)):
        j = s[m]
        sample_document.append(feature_name[j])
    common_function.document(path_document, sample_document)
    time_line = str(e) + "：\t" + str(end - start) + 's\t' + str(sum_path) + '\n'
    common_function.run_time(path_time, time_line)

average_time = total_time / sample.shape[0]
average_path = total_path / sample.shape[0]
average_time_line = '平均运行时间是：' + str(average_time) + 's\n'
average_path_line = '平均路径长度：' + str(average_path)
common_function.run_time(path_time, average_time_line)
common_function.run_time(path_time, average_path_line)


