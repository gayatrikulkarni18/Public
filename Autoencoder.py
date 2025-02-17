#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

#load
(x_train, _), (x_test, _) = mnist.load_data()

#normalize
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_test = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#size
encoding_dim = 32
#input
input_img = Input(shape=(784,))
#encoded
encoded = Dense(encoding_dim, activation='relu')(input_img)
#decoded
decoded = Dense(784, activation='sigmoid')(encoded)
#this model maps its reconstruction
autoencoder = Model(input_img, encoded)
#this model maps its  encoded representation
encoder = Model(input_img, encoded)
#create
encoded_input = Input(shape=(encoding_dim,))
#retrive
decoder_layer =autoencoder.layers[-1]
#create
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
               epoch=50,
               batch_size=256,
               shuffle=True,
               validation_data=(x_test, x_test))
#encode
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
#use
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #display
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




