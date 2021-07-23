# -*- coding: utf-8 -*-
import logging
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#word2vec model
sentences = gensim.models.word2vec.LineSentence('../data/non_seq/q2_non_seq_document.seq')
model = gensim.models.Word2Vec(sentences, hs=1,min_count=1,window=2,size=40)
model.save('../data/non_seq/q2_non_seq_model.model')

# 将获得的词汇对应的向量存放到字典
word_vector_dic = {}
for word in model.wv.index2word:
    word_vector_dic[word] = list(model[word])


with open('../data/non_seq/q2_non_seq_words.vec', 'a') as word_vec_file:
    for key, values in word_vector_dic.items():
        s = str(key)+' '+str(values)+'\n'
        word_vec_file.write(s)
