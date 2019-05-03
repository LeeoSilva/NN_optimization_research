#!/usr/bin/python3

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils  import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.nn import * 
import matplotlib.pyplot as plt
import random
import time
import sys


def variable_summaries(var):
    with tf.name.scope('summaries'):
        mean = tf.reduce_mean(var)
        print(mean) 
        tf.summary.scalar('mean', mean)
        with tf.name.scope('stddeve'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))


def predictionLoop(data):
    while True:
        rand = random.randint(0, data.shape[0]) 
        predictions = new_model.predict(data)
        print(np.argmax(predictions[rand]))
        draw(rand, data)
        time.sleep(1)

def draw(num, image):
    plt.imshow(image[num], cmap=plt.cm.binary) # cmap arguments converts the image to a binary format(grey)
    plt.title('Hand-Written Numbers by: MNIST dataset')
    plt.ylabel('Height')
    plt.xlabel('Width')
    plt.show() 

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--debug": epoch = 1 # Shotens the number of epochs for execution time
    else: epoch = 200 
    image_x, image_y = 28, 28

    mnist = tf.keras.datasets.mnist # hand-written digits 0-9  

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = normalize(x_train, axis=1) # Normalizes all the pixels values to a number from 0-1
    x_test  = normalize(x_test, axis=1) # Normalizes all the pixel values to a number from 0-1 

    model = Sequential() # Feed Forward model
    model.add(Flatten(input_shape=(image_x,image_y))) # Flattens the tensor by converting it to a 1D Tensor 
    model.add(Dense(128, activation=relu)) # REctified Linear Unit activiation function
                                                # Basically if the input is negative it outputs a 0
                                                # If the input is positive it outputs a 1
    model.add(Dense(128, activation=relu)) 
    model.add(Dense(128, activation=softmax)) # Exponential function will increase the probability 
                                              # of maximum value of the previous layer compare to other value

    model.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']) 
    
    model.fit(x_train, y_train, epochs=epoch) 
    val_loss, val_acc = model.evaluate(x_test, y_test)
    tf.summary.scalar(tf.sumary('learning rate'))
    #print(dir(val_loss))
    # print(dir(val_acc))
    #print("Loss: ".format(val_loss.tostring))
    #kprint("Accuracy: ".format(val_acc))

    model.save('MNIST.model')
    new_model = load_model('MNIST.model')
    predictionLoop(x_test)
    
