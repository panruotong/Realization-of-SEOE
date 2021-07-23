# 从excel读入数据存为data
import pandas as pd
import numpy as np

data = pd.read_excel('../data/questionnaire2.xlsx', sheet_name=1)
# data = pd.read_excel('../data/all.xlsx', sheet_name=0)
print(data.head())
print(len(data.index.values))
print(len(data.columns.values))

data_array = np.array(data)
print(data_array.shape[0], data_array.shape[1])
print(data_array)

np.save('../data/q2_label.npy', data_array)




