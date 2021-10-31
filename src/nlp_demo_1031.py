import numpy as np
import json
import tensorflow as tf
import jsonlines
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.device('/gpu:0')

vocab_size = 1000
embedding_dim = 16
max_length = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<oov>'
training_size = 20000
source_data_dir = 'F:\\DL_datasets\\archive(1)\\Sarcasm_Headlines_Dataset.json'
sentences = []
labels = []

with open(source_data_dir,'r+') as f:
    for item in jsonlines.Reader(f):
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
labels = np.array(labels)

training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]
print(type(labels))

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
num_epochs = 30
history = model.fit(training_padded,training_labels,epochs=num_epochs,
                    validation_data=(testing_padded,testing_labels),verbose=2)

import matplotlib.pyplot as plt
def plot_graphs(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('epochs')
    plt.ylabel(string)
    plt.legend([string,'val_'+string])
    plt.show()

plot_graphs(history,'acc')
plot_graphs(history,'loss')