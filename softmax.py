# -*- coding: UTF-8 -*-
#from IPython.display import SVG
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.optimizers import SGD, Adam
#from keras.utils.visualize_util import model_to_dot
from keras.utils import np_utils
import matplotlib.pyplot as plt
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

'''
softmax一般作为神经网络最后一层，作为输出层进行多分类，
Softmax的输出的每个值都是>=0，并且其总和为1，所以可以认为其为概率分布
'''

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
        self.epoch=0
    
    def on_epoch_end(self, epoch, logs=None):
        #pass #pass 不做任何事情，一般用做占位语句。
        print '轮次:',self.epoch ;
        self.epoch=self.epoch+1;
    
    def on_batch_end(self,batch,logs=None):
        logs = logs or {}
        
        #print '批次:',self.batch ;

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

#verbose：训练时显示实时信息，0表示不显示数据，1表示显示进度条，2表示用只显示一个数据
#validation_split：0.2表示20%作为数据的验证集

#model.fit(x_train_1, y_train_1,
#          epochs=20,#nb_epoch=20,
#          verbose=0,
#          batch_size=500,
#          callbacks=[tensorboard, my_tensorboard])

'''
损失函数

损失函数（loss function），是指一种将一个事件（在一个样本空间中的一个元素）映射到一个表达与其事件相关的经济成本或机会成本的实数上的一种函数，在统计学中损失函数是一种衡量损失和错误（这种损失与“错误地”估计有关，如费用或者设备的损失）程度的函数。

交叉熵（cross-entropy）就是神经网络中常用的损失函数。

交叉熵性质：

（1）非负性。

（2）当真实输出a与期望输出y接近的时候，代价函数接近于0.(比如y=0，a～0；y=1，a~1时，代价函数都接近0)。

一个比较简单的理解就是使得 预测值Yi和真实值Y' 对接近，即两者的乘积越大，coss-entropy越小。

交叉熵和准确度变化图像可以看 TensorBoard 。
-------------------------------------------
梯度下降

如果对于所有的权重和所有的偏置计算交叉熵的偏导数，就得到一个对于给定图像、标签和当前权重和偏置的「梯度」，如图所示：
我们希望损失函数最小，也就是需要到达交叉熵最小的凹点的低部。在上图中，交叉熵被表示为一个具有两个权重的函数。

而学习速率，即在梯度下降中的步伐大小。
上面，我们探索了softmax对多分类的支持和理解，知道softmax可以作为一个输出成层进行多分类任务。

但是，这种分类任务解决的都是线性因素形成的问题，对于非线性的，特别是异或问题，如何解决呢？

这时，一种包含多层隐含层的深度神经网络的概念被提出。
'''

#optimizer（优化器），loss（目标函数或损失函数），metrics（评估模型的指标） 
#模型的测试误差指标
#print(model.metrics_names)
# 对测试数据进行测试
#print model.evaluate(x_test_1, y_test_1,
#          verbose=0,
#          batch_size=500);

#['loss', 'acc']
#[0.29414266981184484, 0.97938013374805455],损失函数值和精度值
#x_test_1[1:10],取出10个元素,

#print model.predict_classes(x_test_1[1:10], batch_size=10, verbose=0);

'''
3. 激活函数

激活函数（activation function）可以使得模型加入非线性因素的。

解决非线性问题有两个办法：线性变换、引入非线性函数。

（1）线性变换(linear transformation)

原本一个线性不可分的模型如：X^2 + Y^2 = 1

其图形如下图所示：

'''

fig = plt.figure(0)
degree = np.random.rand(50)*np.pi*2
x_1 = np.cos(degree)*np.random.rand(50)
y_1 = np.sin(degree)*np.random.rand(50)#产生50个圆内点
x_2 = np.cos(degree)*(1+np.random.rand(50))
y_2 = np.sin(degree)*(1+np.random.rand(50))#产生50个圆外点

# x_3 和 y_3 就是切分线
t = np.linspace(0,np.pi*2,50)
#在指定的间隔内返回均匀间隔的数字。
#返回num均匀分布的样本，在[start, stop]。

x_3 = np.cos(t)
y_3 = np.sin(t)



plt.scatter(x_1,y_1,c='red',s=50,alpha=0.4,marker='o')#圆内
plt.scatter(x_2,y_2,c='black',s=50,alpha=0.4,marker='o')#圆外
plt.plot(x_3,y_3)#圆



'''
将坐标轴进行高维变换，横坐标变成X^2，纵坐标变成 Y^2，这是表达式变为了 X + Y = 1，
这样，原来的非线性问题，就变成了一个线性可分的问题，变成了一个简单的一元一次方程了。
'''

fig2 = plt.figure(1)
#令新的横坐标变成x^2,纵坐标变成 Y^2
x_4 = x_1**2
y_4 = y_1**2
x_5 = x_2**2
y_5 = y_2**2

# 这样就可以构建一个一元线性方程进行拟合
x_6 = np.linspace(-1,2,50)
y_6 = 1 - x_6

plt.scatter(x_4,y_4,c='red',s=50,alpha=0.4,marker='o')
plt.scatter(x_5,y_5,c='black',s=50,alpha=0.4,marker='o')
plt.plot(x_6,y_6)
plt.show()