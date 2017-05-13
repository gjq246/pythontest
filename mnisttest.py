# -*- coding: UTF-8 -*-

#mnist神经网络训练，采用LeNet-5模型

import os  
import cv2  
import numpy as np 

from keras.models import Sequential  
from keras.layers import Conv2D, MaxPooling2D, Flatten  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.advanced_activations import PReLU  
from keras.optimizers import SGD, Adadelta, Adagrad
  
from keras.utils import np_utils  
from keras.utils.vis_utils import plot_model  

import h5py 
from keras.models import model_from_json

#读取model  
model = model_from_json(open('my_model_architecture.json').read())  
model.load_weights('my_model_weights.h5')

#读取2张图片测试
testData =  np.empty((2,1,28,28),dtype="float32")
imgfile='./mnisttest/0-71.bmp'
print imgfile
imgData=cv2.imread(imgfile, 0) #数据
arr = np.asarray(imgData,dtype="float32")  

cv2.namedWindow("Image1")   
cv2.imshow("Image1", imgData)  

testData[0,:,:,:] = arr

imgfile='./mnisttest/1-1038.bmp'
print imgfile
imgData=cv2.imread(imgfile, 0) #数据
arr = np.asarray(imgData,dtype="float32")

cv2.namedWindow("Image2")   
cv2.imshow("Image2", imgData)   
  
testData[1,:,:,:] = arr

#转为tensorflow格式
testData = testData.reshape(testData.shape[0], 28, 28, 1)

print model.predict_classes(testData, batch_size=1, verbose=0);

cv2.waitKey (0)  
cv2.destroyAllWindows()  



