import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer  #tokenizer:分词器，产生词典并创建词向量
#要依据单词的编码将句子序列化，得到句子后要对句子进行处理，使得句子编码的长度相同。
from tensorflow.keras.preprocessing.sequence import pad_sequences  #保证编码后 句子长度的一致性

sentences = [
    'I love my dog',  #要用分词器对句子进行编码 从而可以训练
    'I love my cat',
    'you love my dog',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100,oov_token='<00v>')  #选取频率出现最高的100个单词进行编码
tokenizer.fit_on_texts(sentences)   #对于上述句子创建了一个词典
word_index = tokenizer.word_index  #key value         #分词器对大小写不明感，也不关注符号

sequences = tokenizer.texts_to_sequences(sentences) #对句子进行序列化编码
padded = pad_sequences(sequences,padding='post',maxlen=5,truncating='post') #post:从后面开始填充0  maxlen:控制句子编码长度
#当句子长度超过5时，会损失部分信息，truncating='post' 即从句子末尾开始丢失信息

#之后句子会被填充为一个举证
#{'<00v>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
# [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
# [[ 0  0  0  5  3  2  4]
#  [ 0  0  0  5  3  2  7]
#  [ 0  0  0  6  3  2  4]
#  [ 8  6  9  2  4 10 11]]

print(word_index)
print(sequences)
print(padded)
#output:
    #{'my': 1, 'love': 2, 'dog': 3, 'i': 4, 'you': 5, 'cat': 6, 'do': 7, 'think': 8, 'is': 9, 'amazing': 10}
    #[[4, 2, 1, 3], [4, 2, 1, 6], [5, 2, 1, 3], [7, 5, 8, 1, 3, 9, 10]]

# test_data = [
#     'i really love my dog',
#     'my dog loves my manatee'
# ]
# test_seq = tokenizer.texts_to_sequences(test_data)
# print(test_seq)
#output:
#   [[4, 2, 1, 3], [1, 3, 1]]  #因为测试数据中，部分单词未在词典中
#处理方式：当碰到没有见过的词时，输出一个特殊值  oov_token='<00v>
#output:
    #{'<00v>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
    # [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
    # [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]]