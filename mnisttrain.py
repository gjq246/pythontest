# -*- coding: UTF-8 -*-

#mnist神经网络训练，采用LeNet-5模型

import os  
import cv2  
import numpy as np 
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import Conv2D, MaxPooling2D, Flatten  
from keras.optimizers import SGD  
from keras.utils import np_utils  
from keras.utils.vis_utils import plot_model  

def loadData(path,number):  
    data =  np.empty((number,1,28,28),dtype="float32")   #empty与ones差不多原理，但是数值随机，类型随后面设定 
    labels = np.empty((number,),dtype="uint8")   
    listImg = os.listdir(path) 
    count=0 
    for img in listImg:  
       imgData=cv2.imread(path+'/'+img, 0) #数据
       l=int(img.split('-')[0]) #答案
       arr = np.asarray(imgData,dtype="float32")  #将img数据转化为数组形式  
       data[count,:,:,:] = arr   #将每个三维数组赋给data  
       labels[count] = l   #取该图像的数值属性作为标签  
       count=count+1
       print path," loaded ",count
       if count>=number:
          break
    return data, labels  

 
#从图片文件加载数据 
trainData, trainLabels = loadData('./mnisttrain',1000)  
testData, testLabels = loadData('./mnisttest',1000)  
trainLabels = np_utils.to_categorical(trainLabels, 10)
#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数  
testLabels = np_utils.to_categorical(testLabels, 10)  

# tf或th为后端，采取不同参数顺序
#th
#if K.image_data_format() == 'channels_first':
    # -x_train.shape[0]=6000
 #   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    # -x_train.shape:(60000, 1, 28, 28)
  #  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    # x_test.shape:(10000, 1, 28, 28)
    # 单通道灰度图像,channel=1
   # input_shape = (1, img_rows, img_cols)
#else:    #tf
 #   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  #  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
   # input_shape = (img_rows, img_cols, 1)

#tensorflow后端
trainData = trainData.reshape(trainData.shape[0], 28, 28, 1)
testData = testData.reshape(testData.shape[0], 28, 28, 1)

#建立一个Sequential模型  
model = Sequential()  

'''
用于MNIST数据集。
模型输入为32X32的灰度图像，
第一层为6个5X5卷积核，不扩展边界；
第二层为2X2的最大值池化层，步进为2X2；
第三层为16个5X5卷积核，不扩展边界；
第四层为2X2的最大值池化层，步进为2X2；
第五层为展平层，并全连接120个节点；
第六层为全连接层，84个节点；
第七层为全连接softmax层，输出结果。
原论文中第二层池化层和第三层卷积层之间为是部分连接。
http://blog.csdn.net/qqadssp/article/details/70431236
'''
#第一层为6个5X5卷积核，不扩展边界，第一个卷积核要申明input_shape(通道，大小)  ，激活函数采用“tanh”  
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(28,28,1), activation='tanh')) 
 
#第二层为2X2的最大值池化层，步进为2X2；
model.add(MaxPooling2D(pool_size=(2,2)))  

#第三层为16个5X5卷积核，不扩展边界；
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))  

#第四层为2X2的最大值池化层，步进为2X2；
model.add(MaxPooling2D(pool_size=(2,2)))

#第五层为展平层，并全连接120个节点；
model.add(Flatten())   
model.add(Dense(120, activation='tanh')) 
 
#第六层为全连接层，84个节点
model.add(Dense(84, activation='tanh'))  

#第七层为全连接softmax层，输出结果。
model.add(Dense(10, activation='softmax')) 

#参数设置 
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)  
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])  

#开始训练
model.fit(trainData, trainLabels, batch_size=500, epochs=20, verbose=1, shuffle=True)  
  
#输出模型图片
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)  

print model.metrics_names
# 对测试数据进行测试
print model.evaluate(testData, testLabels,
          verbose=0,
          batch_size=500);




