import numpy as np

# 读入feature.name文件 162
f_name = []
with open('../data/q2_feature_name.txt', 'r') as f_name_file:
    for line in f_name_file:
        f_name.append(line.strip())

print(len(f_name))


# 读入样本 (5660,162)
sample = np.load('../data/q2_sample_line.npy')
# sample = np.load('../data/sample_line.npy')

# path
path_document = '../data/non_seq/q2_non_seq_document.seq'
# path_document = '../data/non_seq/non_seq_document.seq'

with open(path_document, 'a') as f:
    for e in range(sample.shape[0]):
        nums = sample[e]
        f_line_seq = ''
        for j in range(len(nums)):
            data = int(nums[j])
            if data == 1:
                f_line_seq = f_line_seq + f_name[j] + ' '
            else:
                continue
        f_line_seq = f_line_seq + '\n'
        f.write(f_line_seq)


