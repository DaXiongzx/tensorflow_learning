import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

tf.device('/gpu:0')
# local_zip = "F:\\DL_datasets\\kagglecatsanddogs_3367a.zip"
# zip_ref = zipfile.ZipFile(local_zip,'r')
# zip_ref.extractall("F:\\DL_datasets")
# zip_ref.close()

# print(len(os.listdir('F:\\DL_datasets\\PetImages\\Cat')))
#
# try:
#     os.mkdir('F:\\DL_datasets\\PetImages\\cats-v-dogs')
#     os.mkdir('F:\\DL_datasets\\PetImages\\cats-v-dogs\\training')
#     os.mkdir('F:\\DL_datasets\\PetImages\\cats-v-dogs\\testing')
#     os.mkdir('F:\\DL_datasets\\PetImages\\cats-v-dogs\\training\\cats')
#     os.mkdir('F:\\DL_datasets\\PetImages\\cats-v-dogs\\training\\dogs')
#     os.mkdir('F:\\DL_datasets\\PetImages\\cats-v-dogs\\testing\\cats')
#     os.mkdir('F:\\DL_datasets\\PetImages\\cats-v-dogs\\testing\\dogs')
#
# except OSError:
#     pass

def split_data(source,training,testing,split_size): #split_size：切分的比例
    files = []
    for filename in os.listdir(source):
        file = source +'\\'+filename
        if os.path.getsize(file) > 0: #过滤图片为空的图片
            files.append(filename)
        else:
            print(filename+'is zero length,so ignoring')

    training_length = int(len(files)*split_size)
    testing_length = int(len(files)-training_length)
    shuffled_set = random.sample(files,len(files))  #在files 里随机采样len(files）个文件--->打乱文件顺序
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = source+'\\'+filename
        destinatin = training+'\\'+filename
        shutil.copyfile(this_file, destinatin)

    for filename in testing_set:
        this_file = source +'\\'+ filename
        destinatin = testing +'\\'+ filename
        shutil.copyfile(this_file, destinatin)

cat_source_dir = "F:\\DL_datasets\\PetImages\\Cat"
training_cat_dir = "F:\\DL_datasets\\PetImages\\cats-v-dogs\\training\\cats"
testing_cat_dir = "F:\\DL_datasets\\PetImages\\cats-v-dogs\\testing\\cats"
dog_source_dir = "F:\\DL_datasets\\PetImages\\Dog"
training_dog_dir = "F:\\DL_datasets\\PetImages\\cats-v-dogs\\training\\dogs"
testing_dog_dir = "F:\\DL_datasets\\PetImages\\cats-v-dogs\\testing\\dogs"

def create_dir(file_dir):
    if os.path.exists(file_dir):
        print('true')
        shutil.rmtree(file_dir)
        os.mkdir(file_dir)
    else:
        os.mkdir(file_dir)

# create_dir(training_cat_dir)
# create_dir(testing_cat_dir)
# create_dir(training_dog_dir)
# create_dir(testing_dog_dir)
#
# split_size = 0.9
# split_data(cat_source_dir,training_cat_dir,testing_cat_dir,split_size)
# split_data(dog_source_dir,training_dog_dir,testing_dog_dir,split_size)
#print(os.listdir(cat_source_dir))

model = tf.keras.models.Sequential([  #LeNet5
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['acc'])


#数据预处理
training_dir = "F:\\DL_datasets\\PetImages\\cats-v-dogs\\training"
validation_dir = "F:\\DL_datasets\\PetImages\\cats-v-dogs\\testing"

train_datagen = ImageDataGenerator(rescale=1.0/255.)  #RGB三层图像中，每层图像的每个像素点的取值范围是0-255  这里处理后取值范围是0-1
training_generator = train_datagen.flow_from_directory(training_dir,batch_size=100,class_mode='binary',target_size=(150,150))  #将尺寸调整为150*150
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(validation_dir,batch_size=100,class_mode='binary',target_size=(150,150))

history = model.fit_generator(training_generator,
                              epochs=20,
                              verbose=1,  #是否要记录训练日志
                              validation_data=validation_generator)

acc = history.history['acc']
val_acc=history.history['val_acc']
loss = history.history['loss']
val_loss=history.history['val_loss']
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

# plt.plot(epochs,acc,'r',"training_acc")
# plt.plot(epochs,val_acc,'b','validation_acc')
# plt.title('Train_Test_acc')
# plt.show()

# plt.plot(epochs,loss,'r',"Training Loss")
# plt.plot(epochs,val_loss,'b',"validation_loss")
plt.show()


# import numpy as np
# from google.colab import files
# from tensorflow.keras.preprocessing import image
#
# uploaded = files.upload()
# for fn in uploaded.keys():
#     path = '/content/'+fn
#     img = image.load_img(path,target_size=(150,150))
#     x = image.img_to_array(img) #得到多维数组形式
#     x = np.expand_dims(x,axis=0)  #拉直
#     images = np.vstack([x]) #将三个通道拉直后拼接
#     classes = model.predict(images,batch_size=10)

