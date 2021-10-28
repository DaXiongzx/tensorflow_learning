from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters


#创建两个数据生成器，指定scaling范围0-1
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

#通过generator可以减少代码量，真实数据的特点：有些图片尺寸不一，需要剪裁 ；数据量比较大，不能一下子装入内存；经常需要修改参数，例如输出尺寸，增补图像拉伸等。
#指向数据训练文件夹
train_generator = train_datagen.flow_from_directory(
"F:\\xldownload\\DL_datasets\\horse-or-human",
    target_size = (300,300), #指定图片输入尺寸
    batch_size = 32,
    class_mode='binary' #指定二分类
)

validation_generator = validation_datagen.flow_from_directory(
"F:\\xldownload\\DL_datasets\\validation-horse-or-human",
    target_size = (300,300),
    batch_size = 32,
    class_mode='binary'
)

hp = HyperParameters()
def build_model(hp):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(hp.Choice('num_filters_layer0',values=[16,64],default=16),(3,3),activation='relu',
                              input_shape=(300,300,3)),
        tf.keras.layers.MaxPool2D(2,2),

       # for i in range(hp.Int("num_conv_layers",1,3,step=1)):  #这里需要用model.add的方法来添加卷积层，这里的作用是调卷积层的层数
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int("hidden_units",128,512,step=32),activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                 optimizer=RMSprop(lr=0.001),
                 metrics=['acc'])
    return model

tuner = Hyperband(
    build_model,
    objective='val_acc',
    max_epochs=15, #预估的训练轮数
    directory='horse_or_human_params'
    hyperParameters=hp,
    project_name = 'my_horse_human_project'
)
tuner.search()  #自动调参训练模型的方式

# history = model.fit(  #原本模型的训练方式
#                    train_generator,  #原来是x_train
#                    epochs=15,
#                    verbose=1,
#                    validation_data = validation_generator,
#                    validation_steps=8
#                    )