#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import random
import os
import cv2
from parameters import *

path = '/Users/akashsharma/desktop/facenet_humanoid/image_database/'

input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
# In[17]:


names = {0:'Fidel_Castro',
 1:'Catherine_Zeta_Jones',
 2:'Jose_Maria_Aznar',
 3:'Bill_Clinton',
 4:'Kofi_Annan',
 5:'David_Beckham',
 6:'Charles_Moose',
 7:'Keanu_Reeves',
 8:'Lance_Armstrong',
 9:'Bill_Gates'}


# In[54]:


def get_input_positive(path1):
    n = len(os.listdir(path1))
    i, j = random.sample(range(1, n-1), 2)
    anchor = cv2.imread(path1 + '/' + str(i) + '.jpg' )
    #print(path1 + '/' + str(i) + '.jpg')
    anchor = cv2.resize(anchor,(96,96))
    anchor = anchor[...,::-1]
    anchor = np.around(np.transpose(anchor, (2,0,1))/255.0, decimals=12)
    positive = cv2.imread(path1 + '/' + str(j) + '.jpg' )
    positive = cv2.resize(positive, (96,96))
    positive = positive[...,::-1]
    positive = np.around(np.transpose(positive, (2,0,1))/255.0, decimals=12)
    return (anchor, positive)

def get_input_negative(path2):
    n = len(os.listdir(path2))
    i, = random.sample(range(1, n-1),1)
    negative = cv2.imread(path2 + '/' + str(i) + '.jpg')
    #print(path2 +'/'+ str(i) + '.jpg')
    negative = cv2.resize(negative, (96, 96))
    negative = negative[...,::-1]
    negative = np.around(np.transpose(negative, (2,0,1))/255.0, decimals = 12)
    return negative

def batch_generator(batch_size = 16):
    y_val = np.zeros((batch_size, 2, 1))
    anchors = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    positives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    negatives = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    while(True):
        for i in range(batch_size):
            j, k = random.sample(range(0,9),2)
            anchors[i], positives[i] = get_input_positive(path + names[j])
            negatives[i] = get_input_negative(path + names[k])
        x_data = {'anchor' : anchors,
                 'anchorPositive' : positives,
                 'anchorNegative' : negatives}
        yield (x_data, [y_val, y_val, y_val])