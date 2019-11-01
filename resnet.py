"""resnet.py

Importing libraries
"""

import tensorflow as tf
from tensorflow.keras import(
    Input,
    Model
)
from tensorflow.keras.layers import(
    Conv2D,
    MaxPooling2D, 
    GlobalAveragePooling2D,
    Activation,
    BatchNormalization, 
    Dense,
    Dropout,
    Add,
    Flatten
)
from tensorflow.keras.utils import to_categorical
import numpy as np

"""Loading data"""

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = tf.cast(x_train, dtype=tf.float32)/255.0
x_test = tf.cast(x_test, dtype=tf.float32)/255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

"""Residual and Identity Block"""

def resnet_layer(input_data, 
                   num_filters = 64, 
                   kernel_size = 3, 
                   strides = 1, 
                   activation = 'relu',  
                   batch_normalization = True):
  '''Function that returns the identity and residual blocks depending upon the 
  current status of the caller function'''

  x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(input_data)
  if batch_normalization:
    x = BatchNormalization()(x)
  if activation is not None:
    x = Activation(activation=activation)(x)
  return x

"""ResNet model"""

def resnet(input_shape, num_blocks):
  num_filters = 64
  num_res_blocks = int(num_blocks/6)

  inputs = Input(shape=input_shape)
  x = resnet_layer(inputs)

  for stage in range(4):
    for block in range(num_res_blocks):
      if stage > 0 and block == 0:
        strides = 2
      else:
        strides = 1
      y = resnet_layer(x, num_filters=num_filters, strides=strides)
      y = resnet_layer(x, num_filters=num_filters, strides=strides, activation=None)

      if stage > 0 and block == 0:
        x = resnet_layer(x, num_filters=num_filters, strides=1, activation=None, batch_normalization=False)
      
      x = Add()([x, y])
      x = Activation('relu')(x)
    
    num_filters *= 2
  
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  y = Flatten()(x)
  output = Dense(10, activation='softmax')(y)

  model = Model(inputs = inputs, outputs = output)
  return model

input_shape = x_train.shape[1:]

model = resnet(input_shape=input_shape, num_blocks=56)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

"""Model Training"""

batch_size = 128
epochs = 100

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

model.evaluate(x_test)
