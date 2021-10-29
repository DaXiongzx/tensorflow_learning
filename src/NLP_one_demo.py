import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer  #tokenizer:分词器，产生词典并创建词向量
#要依据单词的编码将句子序列化，得到句子后要对句子进行处理，使得句子编码的长度相同。

sentences = [
    'I love my dog',  #要用分词器对句子进行编码 从而可以训练
    'I love my cat',
    'you love my dog',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100)  #选取频率出现最高的100个单词进行编码
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index  #key value         #分词器对大小写不明感，也不关注符号
print(word_index)
