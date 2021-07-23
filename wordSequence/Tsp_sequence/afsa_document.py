import numpy as np
from sko.AFSA import AFSA
from wordSequence.Tsp_sequence import functions
import time

# 读入样本、属性
sample = np.load('../data/sample_line.npy')
feature = np.load('../data/feature_line.npy')

# 读入feature.name文件
f_name = [];
with open('../data/feature.name', 'r') as f_name_file:
    for line in f_name_file:
        f_name.append(line.strip())

# path
path_document = '../data/afsa_document.seq'
path_time = '../data/afsa_time.txt'


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
# sample.shape[0]
for e in range(5):
    nums = sample[e]
    feature_name, distance_matrix = functions.feature(nums, f_name, feature)
    num_points = distance_matrix.shape[0]
    # 求最短路径
    start = time.perf_counter_ns()
    # x0：初始解  T_max：初始温度  T_min：终止温度  L：在每个温度下的迭代次数
    afsa = AFSA(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500,
                max_try_num=100, step=0.5, visual=0.3,
                q=0.98, delta=0.5)
    best_x, best_y = afsa.run()
    end = time.perf_counter_ns()
    print(best_x)
    print(best_y)
    print("程序的运行时间是：%s" % (end - start))
    total_time = total_time + (end - start)
    total_path = total_path + best_y
    # 写入document
    sample_document = []
    for m in range(len(best_x)):
        j = best_x[m]
        sample_document.append(feature_name[j])
    print(sample_document)
    functions.document(path_document, sample_document)
    time_line = "第" + str(e) + "次程序的运行时间是：" + str(end - start) + 'ns ' + str((end - start) / 10 ** 9) + 's' + '\n'
    time_line = time_line + "路径总长：" + str(best_y) + '\n'
    functions.run_time(path_time, time_line)

average_time = total_time / 5
average_path = total_path / 5
average_time_line = '平均运行时间是：' + str(average_time) + 'ns ' + str(average_time / 10 ** 9) + 's\n'
average_path_line = '平均路径长度：' + str(average_path)
functions.run_time(path_time, average_time_line)
functions.run_time(path_time, average_path_line)
print('平均运行时间是：' + str(average_time) + 'ns ' + str(average_time / 10 ** 9) + 's')