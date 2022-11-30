import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np
from tensorflow.keras import regularizers

#batch_size = 128

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0
#train_labels = to_categorical(train_labels, 10, dtype ="uint8")
#test_labels = to_categorical(test_labels, 10, dtype ="uint8")
#train_labels = tf.one_hot(train_labels, depth=10)
#test_labels = tf.one_hot(test_labels, depth=10)


x_val = train_data[:10000]
partial_x_train = train_data

y_val =train_labels[:10000]
partial_y_train = train_labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

model.summary()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))"""


vp = [ [10, 64, 0.14, 0.09, "softplus"], [40, 46, 0.3, 0.2, "softplus"]]

for i in range(len(vp)):
    print(vp[i])
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
    model.add(tf.keras.layers.Dense(512, activation='relu', bias_regularizer=regularizers.L2(1e-4)))
    model.add(tf.keras.layers.Dropout(vp[i][2]))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(vp[i][3]))
    model.add(tf.keras.layers.Dense(10, activation=vp[i][4], activity_regularizer=regularizers.L2(1e-5)))
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    history = model.fit(partial_x_train, partial_y_train,epochs=vp[i][0],batch_size=vp[i][1],validation_data=(test_data, test_labels))
