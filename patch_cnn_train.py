# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:50:03 2019

@author: ZHOUFENG
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:26:40 2019

@author: ZHOUFENG
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
#动态分配显存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
sess = tf.Session(config = config)
#os.environ['KERAS_BACKEND']='theano'
import numpy as np
#import scipy

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,Activation
#from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras import regularizers
from keras.utils import np_utils
import matplotlib.pyplot as plt
#import h5py
#from keras.models import load_model

# load Preprocessed data from file
#alltrain_count = 48506#53861#49346#40423#48507#17600#56592#57511
#alltest_count = 25201#35908#49346#40423#32339# 49800#24254#24648
#X_train = np.load("./predata_aug/augsplit_train_48506/X_augsplit100_train_" + "alltrain_count" + str(alltrain_count)  + ".npy")
#y_train = np.load("./predata_aug/augsplit_train_48506/y_augsplit100_train_" + "alltrain_count" + str(alltrain_count) + ".npy")
#X_test = np.load("./predata_aug/augsplit_train_48506/X_augsplit100_test_" + "alltest_count" + str(alltest_count) + ".npy")
#y_test = np.load("./predata_aug/augsplit_train_48506/y_augsplit100_test_" + "alltest_count" + str(alltest_count) + ".npy")

#未进行数据增强，盐城最好的
testRatio = 0.9
#X_train = np.load("./predata/noaug_oversamples/X_stand_pca_train_" + "testRatio" + str(testRatio)  + ".npy")
#y_train = np.load("./predata/noaug_oversamples/y_stand_pca_train_" + "testRatio" + str(testRatio) + ".npy")
#X_test = np.load("./predata/noaug_oversamples/X_stand_pca_test_" + "testRatio" + str(testRatio) + ".npy")
#y_test = np.load("./predata/noaug_oversamples/y_stand_pca_test_" + "testRatio" + str(testRatio) + ".npy")
X_train = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_noover_sameneighbor/alldata_split/X_stand_pca_train_" + "testRatio" + str(testRatio) + ".npy")
y_train = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_noover_sameneighbor/alldata_split/y_stand_pca_train_" + "testRatio" + str(testRatio) + ".npy")
X_test = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_noover_sameneighbor/alldata_split/X_stand_pca_test_" + "testRatio" + str(testRatio) + ".npy")
y_test = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_noover_sameneighbor/alldata_split/y_stand_pca_test_" + "testRatio" + str(testRatio) + ".npy")

#alltrain_count = 2280#1710
#alltest_count = 38290
#X_train = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_sameneighbor/twotypes_split/X_noaugsplit_train_" + "alltrain_count" + str(alltrain_count) + ".npy")
#y_train = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_sameneighbor/twotypes_split/y_noaugsplit_train_" + "alltrain_count" + str(alltrain_count) + ".npy")
#X_test = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_sameneighbor/twotypes_split/X_noaugsplit_test_" + "alltest_count" + str(alltest_count) + ".npy")
#y_test = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_sameneighbor/twotypes_split/y_noaugsplit_test_" + "alltest_count" + str(alltest_count) + ".npy")

# 使用tensorflow作为后端 Reshape data into (numberofsumples, height, width, channels)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]))
X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], X_test.shape[2], X_test.shape[3]))

# 把类标签转化为one-hot编码       convert class labels to on-hot encoding
y_train = np_utils.to_categorical(y_train)#
y_test = np_utils.to_categorical(y_test)

# Define the input shape 
input_shape = X_train[0].shape
print(input_shape)#(30,30,9)

# number of filters
C1 = 16 #32#
pool_size = (2, 2)
# 定义网络框架   Define the model structure
model = Sequential()

model.add(Conv2D(C1, (3, 3), padding = 'same', input_shape = input_shape))#卷积层1 #input:30*30*9 output:30*30*16
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size)) #池化层1 #output:15*15*16
model.add(Dropout(0.15))

model.add(Conv2D(2*C1, (3, 3))) #input:15*15*16 output:13*13*32
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size)) #output:6*6*32
model.add(Dropout(0.15))

model.add(Conv2D(4*C1, (3, 3))) #input:6*6*32 output:4*4*64
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size)) #output:2*2*64
model.add(Dropout(0.25))

model.add(Conv2D(4*C1, (1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(0.25))

model.add(Flatten()) #input:2*2*64 output:256 #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Dense(6*C1,kernel_regularizer = regularizers.l2(0.001)))#全连接层180个神经元 #output:48
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))#全连接2层（2个类别）
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

#plot_model(model,to_file='G:/desktop/myProject/model.png')

# 定义优化和训练方法    Define optimization and train method
#monitor:被监测的量 factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
#patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
#mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少
#epsilon：阈值，用来确定是否进入检测值的“平原区”
#cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
#min_lr：学习率的下限
#verbose：信息展示模式，0或1
reduce_lr = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.9, patience = 5, min_lr = 0.0000001, verbose = 1)

#filepath准备存放模型的地方，
#save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
#checkpointer = ModelCheckpoint(filepath = "./HDF5/checkpoint.hdf5", verbose = 1, save_best_only = True)
#最好的盐城
#checkpointer = ModelCheckpoint(filepath = "./HDF5/checkpoint_noaugrmsprop21_100.hdf5", verbose = 1, save_best_only = True)
checkpointer = ModelCheckpoint(filepath = "./poyang_allfile/poyang_hdf5/patch_hdf5/patch_noaug/noaug_noover_sameneighbor/testratio0.9/checkpoint3_200.hdf5", verbose = 1, save_best_only = True)
#lr:学习率  momentum：动量参数   decay:每次更新后学习率衰减值  nesterov：布尔值，确定是否使用nesterov动量
#sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
#亦称作多类的对数损失，
#注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

history_save = CSVLogger('./poyang_allfile/poyang_history/patch_history/patch_noaug/noaug_noover_sameneighbor/testratio0.9/historysave3_200.csv',separator = ',',append = False)
# 开始训练模型    Start to train model 
history = model.fit(X_train, y_train, 
                    batch_size = 32, 
                    epochs = 200, 
                    verbose = 1, 
                    validation_data = (X_test, y_test),
                    callbacks = [reduce_lr, checkpointer ,history_save],
                    shuffle = True)

# save the model with h5py

#model.save('./model/HSI_model_epochs100.h5')
#最好的盐城
#model.save('./model/HSI_model_epochs100_noaugrmsprop21.h5')
model.save('./poyang_allfile/poyang_model/patch_model/patch_noaug/noaug_noover_sameneighbor/testratio0.9/HSI_model_epochs3_200.h5')

# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.grid(True)
#plt.legend(['train', 'test'], loc = 'upper left') 
#plt.savefig("./result/model_accuracy_100.svg")
#plt.savefig("./result/100_noaug/100_noaugrmsprop/model_accuracy_100_noaugrmsprop21.svg")
#plt.savefig("./yancheng_allfile/yancheng_result/patch_result/patch_noaug/noaug_sameneighbor/testratio0.8/model_accuracy2_200.svg")
#plt.show()

# summarize history for loss 
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.grid(True)
#plt.legend(['train', 'test'], loc = 'upper left') 
#plt.savefig("./result/model_loss_100.svg")
#plt.savefig("./result/100_noaug/100_noaugrmsprop/model_loss_100_noaugrmsprop21.svg")
#plt.savefig("./yancheng_allfile/yancheng_result/patch_result/patch_noaug/noaug_sameneighbor/testratio0.8/model_loss2_200.svg")
#plt.show()
