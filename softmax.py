# -*- coding: UTF-8 -*-
#from IPython.display import SVG
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.optimizers import SGD, Adam
#from keras.utils.visualize_util import model_to_dot
from keras.utils import np_utils
#import matplotlib.pyplot as plt
import tensorflow as tf
#import pandas as pd
import numpy as np
np.random.seed(0)
#设置线程
THREADS_NUM = 20
tf.ConfigProto(intra_op_parallelism_threads=THREADS_NUM)

(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
print('原数据结构：')
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

#数据变换
#分为10个类别
nb_classes = 10

x_train_1 = X_train.reshape(60000, 784)
#x_train_1 /= 255
#x_train_1 = x_train_1.astype('float32')
y_train_1 = np_utils.to_categorical(Y_train, nb_classes)
print('变换后的数据结构：')
#tup2 = (1, 2, 3, 4, 5 );Python的元组与列表类似，不同之处在于元组的元素不能修改。
print(x_train_1.shape, y_train_1.shape)

x_test_1 = X_test.reshape(10000, 784)
y_test_1 = np_utils.to_categorical(Y_test, nb_classes)
print(x_test_1.shape, y_test_1.shape)

'''
原数据结构：
((60000, 28, 28), (60000,)),28*28为图片大小，60000个样本
((10000, 28, 28), (10000,))
变换后的数据结构：
((60000, 784), (60000, 10))，784=28*28
((10000, 784), (10000, 10))
'''

# 构建一个softmax模型
#https://nbviewer.jupyter.org/github/shikanon/MyPresentations/blob/master/DeepLearning/LearnOfDeepLearning.ipynb
# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

model = Sequential()
model.add(Dense(nb_classes, input_shape=(784,)))#全连接，输入784维度, 输出10维度，需要和输入输出对应
model.add(Activation('softmax'))

sgd = SGD(lr=0.005)
#binary_crossentropy，就是交叉熵函数
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#model 概要
model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 10)                7850
_________________________________________________________________
activation_1 (Activation)    (None, 10)                0
=================================================================
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0
_________________________________________________________________
'''

from keras.callbacks import Callback, TensorBoard
import tensorflow as tf


#构建一个记录的loss的回调函数

#类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的第一个参数名称, 按照惯例它的名称是 self。
#self 代表的是类的实例，代表当前对象的地址
#一个类变量，它的值将在这个类的所有实例之间共享。
#class SubClassName (ParentClass1[, ParentClass2, ...]):
#继承语法 class 派生类名（基类名）：

#[]:list列表,数组  ():元组,不可以修改   {}:dictionary字典,key:value
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = [] #清空,self.数据成员：类变量或者实例变量用于处理类及其实例对象的相关的数据

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss')) #把损失函数值加入到self.losses

# 构建一个自定义的TensorBoard类，专门用来记录batch中的数据变化
class BatchTensorBoard(TensorBoard):
    def __init__(self,log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False):
        #super(B, self)首先找到B的父类（就是类A），然后把类B的对象self转换为类A的对象
        super(BatchTensorBoard, self).__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.batch = 0
        self.batch_queue = set()
    
    def on_epoch_end(self, epoch, logs=None):
        pass #pass 不做任何事情，一般用做占位语句。
    
    def on_batch_end(self,batch,logs=None):
        logs = logs or {}
        
        self.batch = self.batch + 1
        
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = float(value)
            summary_value.tag = "batch_" + name
            if (name,self.batch) in self.batch_queue:
                continue
            self.writer.add_summary(summary, self.batch)
            self.batch_queue.add((name,self.batch))
        self.writer.flush()

tensorboard = TensorBoard(log_dir='/home/tensorflow/log/epoch')
my_tensorboard = BatchTensorBoard(log_dir='/home/tensorflow/log/batch')

'''
（1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
（2）iteration：1个iteration等于使用batchsize个样本训练一次；
（3）epoch：1个epoch等于使用训练集中的全部样本训练一次；
'''

model.fit(x_train_1, y_train_1,
          epochs=20,#nb_epoch=20,
          verbose=0,
          batch_size=100,
          callbacks=[tensorboard, my_tensorboard])