from tensorflow import keras
import matplotlib.pyplot as plt
fashion_mnist = keras.datasets.fashion_mnist
(train_imgs,train_labels),(test_imgs,test_labels) = fashion_mnist.load_data()
plt.imshow(train_imgs[0])