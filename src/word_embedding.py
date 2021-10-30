import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import matplotlib.pyplot as plt

tf.device('/gpu:0')

training_dir = 'F:\\DL_datasets\\aclImdb_v1\\aclImdb\\train'
pos_training_dir ='F:\\DL_datasets\\aclImdb_v1\\aclImdb\\train\\pos'
neg_training_dir = 'F:\\DL_datasets\\aclImdb_v1\\aclImdb\\train\\neg'

testing_dir = 'F:\\DL_datasets\\aclImdb_v1\\aclImdb\\test'
pos_testing_dir ='F:\\DL_datasets\\aclImdb_v1\\aclImdb\\test\\pos'
neg_testing_dir = 'F:\\DL_datasets\\aclImdb_v1\\aclImdb\\test\\neg'

# imdb,info = tfds.load('imdb_reviews',with_info=True,as_supervised=True)
# train_data,test_data = imdb['train'],imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

training_text_dir =[]
testing_text_dir=[]

training_pos_list = os.listdir(pos_training_dir)
training_neg_list = os.listdir(neg_training_dir)
testing_pos_list = os.listdir(pos_testing_dir)
testing_neg_list = os.listdir(neg_testing_dir)

for i in training_pos_list:
    training_text_dir.append(pos_training_dir+"\\"+i)
    training_labels.append(1)

for i in training_neg_list:
    training_text_dir.append(neg_training_dir+"\\"+i)
    training_labels.append(0)
#print(len(training_pos_list))

for i in testing_pos_list:
    testing_text_dir.append(pos_testing_dir+"\\"+i)
    testing_labels.append(1)

for i in testing_neg_list:
    testing_text_dir.append(neg_testing_dir+"\\"+i)
    testing_labels.append(0)

for path in training_text_dir:
    with open(path,'r',encoding='utf-8') as f:
        training_sentences.append(f.read())
for path in testing_text_dir:
    with open(path,'r',encoding='utf-8') as f:
        testing_sentences.append(f.read())



training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
# print(training_labels_final)
#
vocab_size = 10000  #词典长度为10000
embedding_dim = 16  #每个单词被表示在16维的空间中
max_length = 120  #每个句子最多120个单词
trunc_type = 'post'
oov_tok = '<oov'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words= vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences) ##
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)


#有一部分词在testing_datasets中但不在training_data中
test_sequences = tokenizer.texts_to_sequences(testing_sentences)##所以是用训练数据集产生的词典来进行编码和序列化
testting_padded = pad_sequences(test_sequences,maxlen=max_length) #有些测试数据中的单词没有出现在训练集中，所以会出现oov
                                                                #可以检测神经网络对未遇到词的预测能力

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),#word_embedding 将单词映射到矢量空间
    tf.keras.layers.Flatten(),
    #tf.keras.layers.GlobalAveragePooling1D(),#在每个向量的维度上取平均值输出
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
#model.summary()
num_epochs = 10
model.fit(padded,training_labels_final,epochs=num_epochs,validation_data=(testting_padded,testing_labels_final))

e = model.layers[0]
print(e)
weights = e.get_weights()[0] #固定写法
print(weights.shape)

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])  #将词典中的key value反转

import io
out_v = io.open('vecs.tsv','w',encoding='utf-8')  #embedding
out_m = io.open('meta.tsv','w',encoding='utf-8')  #对应的词
for word_num in range(1,vocab_size):
    word = reverse_word_index[word_num]  #第几个单词
    embeddings = weights[word_num]  #第几个单词对应的词向量，词向量存放在weights中，weights是10000*16矩阵
    out_m.write(word+'\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
out_v.close()
out_m.close()