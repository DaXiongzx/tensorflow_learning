import tensorflow as tf
from tensorflow import keras
import numpy as np


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss')<0.4):
            print("stop training")
            self.model.stop_training = True


callbacks = myCallback()

fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# model.summary() #中间层的每个neuron都会有个bias

train_imgs = train_imgs / 255  # fashion mnist 28*28 每个像素点为灰度值（0，255） 做了个一次normalization/scaling的处理，让数值处于0-1之间，方便训练
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])  # metrics=['accuracy']看到精度信息
# output_label 是one hot vector
model.fit(train_imgs, train_labels, epochs=5, callbacks=[callbacks])
# 训练数据集做了normalization,所以测试数据集也要做normalization
# test_imgs_scaled = test_imgs / 255
# model.evaluate(test_imgs_scaled, test_labels)
# print(np.argmax(model.predict([[test_imgs[0] / 255]])))
# print(test_labels[0])
# import matplotlib.pyplot as plt
#
# plt.imshow(test_imgs[0])