# -*- coding: UTF-8 -*-

#import cv2  
#from keras.datasets import mnist

#import numpy as np  
#i=4
#s='aa'+str(4)
#print s
#(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

#for i in range(0,59999):  # 迭代 10 到 20 之间的数字
#   fileName="./mnisttrain/"+str(Y_train[i])+"-"+str(i)+".bmp"
#   cv2.imwrite(fileName, X_train[i])

#for i in range(0,9999):  # 迭代 10 到 20 之间的数字
#   fileName="./mnisttest/"+str(Y_test[i])+"-"+str(i)+".bmp"
#   cv2.imwrite(fileName, X_test[i])

#print X_train[0]
#cv2.imwrite("./mnisttrain/1.bmp", X_train[i])
#print Y_train[0]

#img = cv2.imread("./mnisttrain/1.bmp",0)#0:gray 
#print img

import os  
import cv2  
from numpy import *  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import Conv2D, MaxPooling2D, Flatten  
from keras.optimizers import SGD  
from keras.utils import np_utils  
from keras.utils.vis_utils import plot_model  

def loadData(path):  
    data = []  
    labels = []  
    listImg = os.listdir(path) 
    count=0 
    for img in listImg:  
       data.append([cv2.imread(path+'/'+img, 0)]) 
       l=int(img.split('-')[0]) 
       labels.append(l)  
       print path, l, 'is read' 
       count=count+1
       if count>1000:
          break
    return data, labels  
  
trainData, trainLabels = loadData('./mnisttrain')  
testData, testLabels = loadData('./mnisttest')  
trainLabels = np_utils.to_categorical(trainLabels, 10)
#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数  
testLabels = np_utils.to_categorical(testLabels, 10)  
#trainData = trainData.reshape(trainData.shape[0], 28, 28, 1)
#testData = testData.reshape(testData.shape[0], 28, 28, 1)

model = Sequential()  
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(28,28,1), activation='tanh')) 
#model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=(1,28,28)))  
#model.add(Activation('tanh'))  
 
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Flatten())  
model.add(Dense(120, activation='tanh'))  
model.add(Dense(84, activation='tanh'))  
model.add(Dense(10, activation='softmax'))  
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])  
model.fit(trainData, trainLabels, batch_size=500, epochs=20, verbose=1, shuffle=True)  
  
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)  

print model.metrics_names
# 对测试数据进行测试
print model.evaluate(testData, testLabels,
          verbose=0,
          batch_size=500);


