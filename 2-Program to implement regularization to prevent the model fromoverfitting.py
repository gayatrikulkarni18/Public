#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf # import Tensorflow libraray

#Load the data
(train_data,train_labels), (test_data,test_labels) = tf.keras.datasets.mist.load_data()#load
#preprocess the data
train_data = train_data.reshape((60000, 784)) / 255.0 # reshape and normalize training data
test_data = test_dat.reshape((10000, 784)) / 255.0 #reshape
train_labels = tf.keras.utlise.to_categorical(train_labels) # convert training
test_labels = tf.keras.utlis.to_categorical((test_labels)) #convert testing
test_labels = tf.keras.utils.to_categorical(test_labels) # convert testing

#define 
model = tf.keras.models.Sequential([ #define sequential model
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,), kernal_regularizer=tf.keras.regularizers.l2(0.01)), #add 
    tf.keras.layers.Dense(64, activation='relu', kernal_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(64, activation='softmax')
])
#compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
#train
history = model.fit(train_data, train_labels, epochs=10, batch_sizes=128, validation_data=(test_data, test_labels))




