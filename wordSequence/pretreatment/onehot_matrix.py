import numpy as np


# 读入pre_data.data文件，str转int
def readfile():
    # 打开文件（注意路径）
    f = open('../data/q2_pre_data.data', 'r')
    # f = open('../data/pre_data.data', 'r')
    # 逐行进行处理
    first_ele = True
    for data in f.readlines():
        ## 去掉每行的换行符，"\n"
        data = data.strip()
        ## 按照 制表符进行分割。
        nums = data.split("\t")
        ## 添加到 matrix 中。
        if first_ele:
            ### 将字符串转化为整型数据
            nums = [int(x) for x in nums]
            ### 加入到 matrix 中 。
            matrix = np.array(nums)
            first_ele = False
        else:
            nums = [int(x) for x in nums]
            matrix = np.c_[matrix, nums]
    f.close()
    return matrix


# 矩阵转置
def dealMatrix(matrix):
    ## 一些基本的处理。
    matrix = matrix.transpose()

    return matrix


# 保存转置矩阵、距离矩阵
feature = readfile()
# (feature, sample) (162, 5660)
print(feature.shape)
np.save('../data/q2_feature_line.npy', feature)
# np.save('../data/feature_line.npy', feature)

sample = dealMatrix(feature)
# print("transpose the matrix")
# (sample, feature) (5660, 162)
print(sample.shape)
np.save('../data/q2_sample_line.npy', sample)
# np.save('../data/sample_line.npy', sample)

