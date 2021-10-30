import json
import jsonlines
from tensorflow.keras.preprocessing.text import Tokenizer  #tokenizer:分词器，产生词典并创建词向量
#要依据单词的编码将句子序列化，得到句子后要对句子进行处理，使得句子编码的长度相同。
from tensorflow.keras.preprocessing.sequence import pad_sequences  #保证编码后 句子长度的一致性


source_file = 'F:\\DL_datasets\\archive(1)\\Sarcasm_Headlines_Dataset.json'
article_link_list =[]
sentences = []
labels = []
with open(source_file,'r+') as f:
    for item in jsonlines.Reader(f):
        article_link_list.append(item['article_link'])
        labels.append(item['is_sarcastic'])
        sentences.append(item['headline'])

tokenizer = Tokenizer(oov_token='<oov')
tokenizer.fit_on_texts(sentences)  #初始化分词器 并且生成单词编码
word_index = tokenizer.word_index  #可以获得对单词的编码
#print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)   #用生成的单词编码对句子进行编码
padded = pad_sequences(sequences,padding='post')  #最大文章标题句子长度为40个单词  所以对每个句子的编码都是40个单词 padded用来填充长度不够的句子
print(padded[0:10])
print(padded.shape)

# print("article_link")
# print(article_link_list)
# print("\r")
# print("labels")
# print(labels)
# print("\r")
# print("sentences")
# print(sentences)

