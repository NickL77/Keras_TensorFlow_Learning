#!/usr/bin/python
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

dirname = os.path.realpath('.') + '/mnist_num_reader.h5'

#import mnist dataset and normalize pixels to grayscale (0-1)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

#build model
model = tf.keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

#add training parameters and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

#calulate validation accuracy and loss
val_loss, val_acc = model.evaluate(x_test, y_test)
print('val_loss', val_loss)
print('val_acc', val_acc)

#save model
keras.models.save_model(model, dirname)

#load in trained model
new_model = keras.models.load_model('mnist_num_reader.h5', compile=False)

#run model on test set (choose max value for identifying digit)
predictions = [np.argmax(pred) for pred in new_model.predict(x_test)]

#print an example test
print(predictions[1])
plt.imshow(x_test[1],cmap=plt.cm.binary)
plt.show()
