import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
#10 labels
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

train_images = train_images/225.0
test_images = test_images/225.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    #128 each is connected to the every other neuron
    #relu is rectefied linear unit
    keras.layers.Dense(10,activation='softmax')
    #softmax picks value for each neuron so all the values add up to 1
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#try to mess with different loss functions, look up what's adam

model.fit(train_images, train_labels, epochs=10)
#epochs - how many times will the model see information
#won't neccessery increase the accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Tested accuracy: ', test_acc)
