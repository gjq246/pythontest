# -*- coding: UTF-8 -*-

#把mnist数据集转成图片做测试，图片更为通用

import cv2  
from keras.datasets import mnist

import numpy as np  

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

for i in range(0,59999):  # 迭代 0 到 59999 之间的数字
   fileName="./mnisttrain/"+str(Y_train[i])+"-"+str(i)+".bmp"
   cv2.imwrite(fileName, X_train[i])

for i in range(0,9999):  # 迭代 0 到 9999 之间的数字
   fileName="./mnisttest/"+str(Y_test[i])+"-"+str(i)+".bmp"
   cv2.imwrite(fileName, X_test[i])

