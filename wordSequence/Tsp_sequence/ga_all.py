import numpy as np
import time

# # 保存转置矩阵、距离矩阵
# feature = np.load('../data/feature_line.npy')
#
# # 计算距离矩阵(hamming)
# distance_matrix = np.zeros((feature.shape[0], feature.shape[0]))
# for i in range(feature.shape[0]):
#     for j in range(feature.shape[0]):
#         non = np.nonzero(feature[j] - feature[i])
#         distance_matrix[i, j] = len(non[0])
#
# np.save('../data/all_feature_distance.npy', distance_matrix)

distance_matrix = np.load('../data/all_feature/all_feature_distance.npy')


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %% do GA
from sko.GA import GA_TSP

start = time.time()
ga_tsp = GA_TSP(func=cal_total_distance, n_dim=distance_matrix.shape[0], size_pop=50, max_iter=500, prob_mut=0.2)
best_points, best_distance = ga_tsp.run()
end = time.time()
print(best_points)
print(best_distance)
print("程序的运行时间是：%s" % (end - start))
np.save('../data/all_feature/all_feature_sequence.npy', best_points)

# 读入feature.name文件 162
f_name = []
with open('../data/feature.name', 'r') as f_name_file:
    for line in f_name_file:
        f_name.append(line.strip())

# 读入样本 (5660,162)
sample = np.load('../data/sample_line.npy')

# path
path_document = '../data/all_feature/all_feature_sequence.seq'
with open(path_document, 'a') as f:
    for e in range(sample.shape[0]):
        nums = sample[e]
        f_line_seq = ''
        for j in range(len(nums)):
            index = best_points[j]
            data = int(nums[index])
            if data == 1:
                f_line_seq = f_line_seq + f_name[index] + ' '
            else:
                continue
        f_line_seq = f_line_seq + '\n'
        f.write(f_line_seq)

