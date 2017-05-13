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
  
#model.add(Conv2D(4, 5, 5, border_mode='valid',input_shape=(28,28,1)))    
#第一个卷积层，4个卷积核，每个卷积核5*5,卷积后24*24，第一个卷积核要申明input_shape(通道，大小) ,激活函数采用“tanh”  
model.add(Conv2D(filters=4, kernel_size=(5,5), padding='valid', input_shape=(28,28,1), activation='tanh')) 
 
  
#model.add(Conv2D(8, 3, 3, subsample=(2,2), border_mode='valid'))   
#第二个卷积层，8个卷积核，不需要申明上一个卷积留下来的特征map，会自动识别，下采样层为2*2,卷完且采样后是11*11  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(filters=8, kernel_size=(3,3), padding='valid', activation='tanh'))
#model.add(Activation('tanh'))  
  
#model.add(Conv2D(16, 3, 3, subsample=(2,2), border_mode='valid'))   
#第三个卷积层，16个卷积核，下采样层为2*2,卷完采样后是4*4  
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='valid', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))  
#model.add(Activation('tanh'))  
  
model.add(Flatten())   
#把多维的模型压平为一维的，用在卷积层到全连接层的过度
  
#model.add(Dense(128, input_dim=(16*4*4), init='normal'))   
#全连接层，首层的需要指定输入维度16*4*4,128是输出维度，默认放第一位 
model.add(Dense(128, activation='tanh'))  

#model.add(Activation('tanh'))  
  
#model.add(Dense(10, input_dim= 128, init='normal'))   
#第二层全连接层，其实不需要指定输入维度，输出为10维，因为是10类 
model.add(Dense(10, activation='softmax')) 
#model.add(Activation('softmax'))   
#激活函数“softmax”，用于分类  
  
#训练CNN模型   
  
sgd = SGD(lr=0.05, momentum=0.9, decay=1e-6, nesterov=True)   
#采用随机梯度下降法，学习率初始值0.05,动量参数为0.9,学习率衰减值为1e-6,确定使用Nesterov动量  
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])   
#配置模型学习过程，目标函数为categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列，第18行已转化，优化器为sgd  

model.fit(trainData, trainLabels, batch_size=100,epochs=20,shuffle=True,verbose=1,validation_split=0.2)   
#训练模型，训练nb_epoch次，bctch_size为梯度下降时每个batch包含的样本数，验证集比例0.2,verbose为显示日志，shuffle是否打乱输入样本的顺序  

#输出模型图片
plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=False)  

print model.metrics_names
# 对测试数据进行测试
print model.evaluate(testData, testLabels,
          verbose=0,
          batch_size=500);

#保存model
json_string = model.to_json()  
open('my_model_architecture.json','w').write(json_string)  
model.save_weights('my_model_weights.h5') 



