import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from scipy.misc import imread
from random import shuffle

import urllib

results = '...'
dir_no = 'data/after/data/sorted_100/no/'
dir_min = 'data/after/data/sorted_100/min/'
dir_sig = 'data/after/data/sorted_100/sig/'
dir_dest = 'data/after/data/sorted_100/dest/'

#get full images list 
no_names = os.listdir(dir_no)
min_names = os.listdir(dir_min)
sig_names = os.listdir(dir_sig)
dest_names = os.listdir(dir_dest)

print("image count:", len(no_names) + len(min_names) + len(sig_names) + len(dest_names))

#read in images
no_images = [imread(dir_no + name) for name in no_names]
min_images = [imread(dir_min + name) for name in min_names]
sig_images = [imread(dir_sig + name) for name in sig_names]
dest_images = [imread(dir_dest + name) for name in dest_names]

print("image read")

#these are important for the NN
IMG_SIZEX, IMG_SIZEY, NLAYERS = no_images[0].shape

print("Image shape:", no_images[0].shape)

#make labels for the CNN
no_labels = [[1,0,0,0] for x in range(0,len(no_images))]
min_labels = [[0,1,0,0] for x in range(0,len(min_images))]
sig_labels = [[0,0,1,0] for x in range(0,len(sig_images))]
dest_labels = [[0,0,0,1] for x in range(0,len(dest_images))]

#make lists that contain both the image, and its associated label
train_data_no = [[no_images[x], no_labels[x]] for x in range(0,len(no_labels))]
train_data_min = [[min_images[x], min_labels[x]] for x in range(0,len(min_labels))]
train_data_sig = [[sig_images[x], sig_labels[x]] for x in range(0,len(sig_labels))]
train_data_dest = [[dest_images[x], dest_labels[x]] for x in range(0,len(dest_labels))]

#concatenate the arrays, and shuffle the data
train_data = train_data_no + train_data_min + train_data_sig + train_data_dest
shuffle(train_data)

print("training data ready, setting up NN")

"""
# Simple neural network architecture

convnet = input_data(shape=[None, 256, 256, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.1, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

"""
# Paper (Vetrivel et al 2017) neural network architecture

convnet = input_data(shape=[None, 100, 100, 3], name='input')

convnet = conv_2d(convnet, 9, 11, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 21, 7, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 41, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 256, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 100, activation='softmax')
convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

#Train the network

print("start training and outputting results")

perc = int(len(train_data) * 0.85)
print(perc)
train = train_data[:perc]
test = train_data[perc:]

NX, NY = 100, 100

X = np.array([i[0] for i in train]).reshape(-1,NX,NY,3)
Y = [i[1] for i in train]

print(X[0].shape)

test_x = np.array([i[0] for i in test]).reshape(-1,NX,NY,3)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=99, show_metric=True) #, run_id="test_run")
