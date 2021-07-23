# -*- coding: utf-8 -*-
import logging
import gensim
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# load word2vec model
model = gensim.models.Word2Vec.load('../data/sa/q2_sa_model.model')
# model = gensim.models.Word2Vec.load('../data/ia/q2_ia_model.model')
# model = gensim.models.Word2Vec.load('../data/ga/q2_ga_model.model')
# model = gensim.models.Word2Vec.load('../data/greedy/q2_greedy_model.model')
# model = gensim.models.Word2Vec.load('../data/non_seq/q2_non_seq_model.model')
# model = gensim.models.Word2Vec.load('../data/ga/ga_model.model')
# model = gensim.models.Word2Vec.load('../data/all_feature/all_feature_model.model')
# model = gensim.models.Word2Vec.load('../data/non_seq/non_seq_model.model')
# model = gensim.models.Word2Vec.load('../data/greedy/greedy_model.model')
# model = gensim.models.Word2Vec.load('../data/ia/ia_model.model')
# model = gensim.models.Word2Vec.load('../data/sa/sa_model.model')

print("和长过痤疮相似的属性：")
for key in model.most_similar('Q8_1', topn=10):
    print(key)

print("和没长过痤疮相似的属性：")
for key in model.most_similar('Q8_2', topn=10):
    print(key)

# print("和抑郁相似的属性：")
# for key in model.most_similar('p33', topn=10):
#     print(key)
#
# print("和焦虑相似的属性：")
# for key in model.most_similar('p34', topn=10):
#     print(key)
#
# print("和焦虑、抑郁相似的属性：")
# print(model.most_similar(positive=['p33', 'p34'], topn=10))
#
# print("和健康相似的属性：")
# for key in model.most_similar('p41', topn=10):
#     print(key)


# dic = {}
# with open('../data/feature.describe', 'r', encoding='UTF-8') as f:
#     for line in f.readlines():
#         line.strip()
#         list = line.split("\t")
#         print(list)
#         # dic[list[0]] = list[1]
#
# print(len(dic))
# print('dic[\'p02_1\']', dic['p02_1'])
# np.save('../data/feature_dict.npy', dic)


