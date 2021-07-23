# 从excel读入数据存为data
import pandas as pd
import numpy as np

data = pd.read_excel('../data/questionnaire2.xlsx', sheet_name=0)
# data = pd.read_excel('../data/all.xlsx', sheet_name=0)
print(data.head())
print(len(data.index.values))
print(len(data.columns.values))

data_array = np.array(data)
print(data_array.shape[0], data_array.shape[1])
print(data_array[0])

# 样本的one-hot编码
with open('../data/q2_pre_data.data', 'w') as all_feature:
    for i in range(data_array.shape[0]):
        data_line = ''
        for j in range(data_array.shape[1]):
            value = str(int(data_array[i][j]))
            data_line = data_line + value +'\t'
        data_line = data_line+'\n'
        all_feature.write(data_line)
        # print(data_line)




