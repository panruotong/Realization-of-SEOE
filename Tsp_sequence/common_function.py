import numpy as np


# 输入：样本 输出：样本包含的属性、属性的距离矩阵
def feature(nums, f_name, feature):
    feature_name = [];
    first_one = True
    for j in range(len(nums)):
        data = int(nums[j])
        if (data == 1) and first_one:
            matrix = np.array(feature[j])
            feature_name.append(f_name[j])
            first_one = False
        elif (data == 1) and (not first_one):
            matrix = np.c_[matrix, feature[j]]
            feature_name.append(f_name[j])
        else:
            continue
    matrix = matrix.transpose()
    # 计算距离矩阵(hamming)
    distance_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            non = np.nonzero(matrix[j] - matrix[i])
            distance_matrix[i, j] = len(non[0])
    return feature_name, distance_matrix


# 输入：属性序列 输出：document文件
# path：文件路径
def document(path, sample_document):
    with open(path, 'a') as f_seq_file:
        f_line_seq = ''
        for x in range(len(sample_document)):
            f_line_seq = f_line_seq + sample_document[x]+' '
        f_line_seq = f_line_seq + '\n'
        f_seq_file.write(f_line_seq)


def run_time(path, line):
    with open(path, 'a', encoding='utf-8') as runtime:
        runtime.write(line)


def out():
    print('调用成功')
