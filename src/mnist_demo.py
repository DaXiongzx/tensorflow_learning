import csv
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop

tf.device('/gpu:0')


def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        first_line = True
        temp_imgs = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                temp_labels.append(row[0])
                img_data = row[1:785]
                img_data_as_array = np.array_split(img_data, 28)
                temp_imgs.append(img_data_as_array)

        imgs = np.array(temp_imgs).astype('float')
        labels = np.array(temp_labels).astype('float')
    return imgs, labels

training_data_file_path = 'F:\\DL_datasets\\archive\\sign_mnist_train.csv'
testing_data_file_path = 'F:\\DL_datasets\\archive\\sign_mnist_test.csv'

training_imgs, training_labels = get_data(training_data_file_path)
testing_imgs, testing_labels = get_data(testing_data_file_path)
#print(training_imgs.shape)

#数据预处理部分----------------------------------------
training_imgs = np.expand_dims(training_imgs,axis=3)
testing_imgs = np.expand_dims(testing_imgs,axis=3)

train_datagen = ImageDataGenerator(
    rescale =1./255,
    rotation_range = 40,  #data augmentation 图片旋转四十度
    width_shift_range = 0.2,#data augmentation
    height_shift_range=0.2,#data augmentation
    shear_range=0.2,#data augmentation
    zoom_range=0.2,#data augmentation
    horizontal_flip = True,#data augmentation  平移
    fill_mode = 'nearest' #平移后对数据空白的填充
)
validation_datagen = ImageDataGenerator(rescale = 1./255)

#-------------------------------------over----------------------

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(26,activation=tf.nn.softmax)
])
model.compile(optimizer=RMSprop(lr=0.001),
              loss = 'sparse_categorical_crossentropy', #multi-class
              metrics=['acc'])

history = model.fit_generator(train_datagen.flow(training_imgs,training_labels,batch_size=32),
                              steps_per_epoch=len(training_imgs)/32,
                              epochs=30,
                              verbose=1,
                              validation_data=validation_datagen.flow(testing_imgs,testing_labels,batch_size=32),
                              validation_steps=len(testing_imgs)/32)
#model.evaluate(testing_imgs,testing_labels)ACC

acc = history.history['acc']   #会保留一个epoch中最后一个step的acc
val_acc=history.history['val_acc']
loss = history.history['loss']
val_loss=history.history['val_loss']
print('len(acc):{}'.format(acc))

epochs = range(len(acc))


ax1 = plt.subplot(2,2,1)
#第一行第二列图形
ax2 = plt.subplot(2,2,2)

plt.sca(ax1)
plt.plot(epochs,acc,'r',"training_acc")
plt.plot(epochs,val_acc,'b','validation_acc')


# 选择ax2
plt.sca(ax2)
plt.plot(epochs,loss,'r',"Training Loss")
plt.plot(epochs,val_loss,'b',"validation_loss")

plt.show()