import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPool2D(2,2), #原有的输出会小一半
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
fashion_mnist = keras.datasets.fashion_mnist
(train_imgs,train_labels),(test_imgs,test_labels) = fashion_mnist.load_data()

train_imgs = train_imgs/255
test_imgs = test_imgs/255

# model.compile(optimizer = tf.optimizers.Adam(),loss = tf.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
# model.fit(train_imgs.reshape(-1,28,28,1),train_labels,epochs=1)#要将输入数据reshape成4维
# print("-------------------------------")


layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.models.Model(inputs = model.input,outputs = layer_outputs) #model.input就是输入层 https://blog.csdn.net/qq_31112205/article/details/103046794
pred = activation_model.predict(test_imgs[0].reshape(1,28,28,1))