import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#load
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

#preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_test, -1)

#define 
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, (3,3), activation="relu"),
    keras.layers.Dense(64, (3,3), activation="softmax"),
])

#Compile
model.compile(optmizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#train
history = model.fit(x_train, y_train, epochs=l5, batch_sizes=128, validation_data=(x_test, y_test))

#evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

#show
sample_img = x_test[0]
sample_label = y_test[0]
sample_img = np.predict(sample_img, 0)
pred = model.predict(sample_img)
pred_label = np.argmax(pred)
print("Sample image true label:", sample_label)
print("Sample image predicted label:", sample_label)


#display
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.show()

