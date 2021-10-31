import numpy as np
import json
import tensorflow as tf
import jsonlines
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."
corpus = data.lower().split('\n')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus) #创建了词典
total_word = len(tokenizer.word_index)+1
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]  #line:in the town of athy one jeremy lanigan   所以要[line]组成一个列表 固定写法
    #print(token_list)
    for i in range(1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        #[1, 26]  递增式的输出序列
        # [1, 26, 61]
        # [1, 26, 61, 60]
        # [1, 26, 61, 60, 262]
        # [1, 26, 61, 60, 262, 13]
        # [1, 26, 61, 60, 262, 13, 9]
        # [1, 26, 61, 60, 262, 13, 9, 10]
        #此时相当于得到一个新的句子列表，对列表里的所有句子进行padded
max_sequence_len = max(len(x) for x in input_sequences)
#print(input_sequences)
input_sequences = np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))
xs = input_sequences[:,:-1] #第一个：指的是取所有行的数据
labels = input_sequences[:,-1] #取所有行的最后一个数据

ys = tf.keras.utils.to_categorical(labels,num_classes=total_word) #转换成one_hot_vector 词典中的每个单词都有可能被预测

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_word,64,input_length=max_sequence_len-1),#total_word：需要处理的单词数量为语料库中所有单词
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)), #20 means cell state所处理的上下文长度
    tf.keras.layers.Dense(total_word,activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
epochs = 500
history = model.fit(xs,ys,epochs=500,verbose=1)


next_words = 10
seed_sentences = 'Laurence went to dublin'
for i in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_sentences])[0]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict_classes(token_list,verbose=0)
    output_word = ''
    for word,index in tokenizer.word_index.items():
        if index==predicted:
            output_word = word
            break
    seed_sentences+=' '+output_word
print(seed_sentences)
acc = history.history['acc']
plt.plot(acc)
# plt.plot([y for y in range(epochs)])
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()