# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:22:02 2019

@author: ZHOUFENG
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:32:21 2019

@author: ZHOUFENG
"""

import numpy as np
#import gdal
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import  StratifiedShuffleSplit
import os
import scipy.io as io
from sklearn.model_selection import train_test_split

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
        return
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_bands = dataset.RasterCount #波段数
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
    #im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
    #im_proj = dataset.GetProjection()#获取投影信息
    mat1 = np.zeros(((im_height,im_width,im_bands)))
    for i in range(0,im_bands):
        mat1[:,:,i] = im_data[i,0:im_height,0:im_width]
    return mat1


# 切分训练集和测试集，从全部的训练数据中洗牌10次随机选取
def splitTrainTestSet(X, y,classnum = 10,  testRatio = 0.80):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                test_size=testRatio, random_state=345, stratify=y)
    ss = StratifiedShuffleSplit(n_splits = classnum, test_size = testRatio, train_size = 1 - testRatio, random_state = 0)
    #交叉验证共10次，每次训练样本和测试样本对分
    for train_index, test_index in ss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)#获取索引值
        X_train, X_test = X[train_index], X[test_index]#获取索引对应的具体的数据
        y_train, y_test = y[train_index], y[test_index]#获取索引对应的具体的标签值
        
    return X_train, X_test, y_train, y_test


#保存预处理数据文件      save Preprocessed Data to file
def savePreprocessedData(path, X_trains, X_tests, y_trains, y_tests, testRatio = 0.8):
    
    data_path = os.path.join(os.getcwd(), path)
    with open(os.path.join(data_path, "X_stand_pca_train_") + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
        np.save(outfile, X_trains)
    with open(os.path.join(data_path, "X_stand_pca_test_") + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
        np.save(outfile, X_tests)     
    with open(os.path.join(data_path, "y_stand_pca_train_") + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
        np.save(outfile, y_trains)     
    with open(os.path.join(data_path, "y_stand_pca_test_") + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
        np.save(outfile, y_tests)

#重采样
def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts  
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y== label,:,:,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    return newX, newY
    

#Samples_Path  = 'F:\\b705\\ZHOUFENG\\change\\change\\Samples_truelabel\\'
def LoadSamples():
    #Samples_Path  = 'F:\\b705\\ZHOUFENG\\change\\change\\river_predata\\patchdata\\Samples_pca_patch_truelabel_sameneighbor\\'       
    Samples_Path  = 'D:\\Poyang\\matsamples\\patchdata\\Samples_pca_patch_truelabel_sameneighbor\\'       
    Samples = np.zeros((((40000,30,30,9))))
    index = 0
    #testRatio = 0.3
    Label = np.zeros(40000,dtype = np.int)
    for i in range(1,40001):
        PSample_mat = Samples_Path + 'Samples_'+str(i) + '_1.mat'
        NSample_mat = Samples_Path + 'Samples_'+str(i) + '_0.mat'
        
        try:
            PMat = io.loadmat(PSample_mat).get('k')
            #print('正例')
            Samples[index,:,:,:] = PMat
            Label[index] = 1
            index = index + 1
           # print(PMat)
        except:
            NMat = io.loadmat(NSample_mat).get('k')
            #print('反例')
            Samples[index,:,:,:] = NMat
            Label[index] = 0
            index = index + 1
    return Samples,Label

#44100,18900
#def Main():
Samples,Label = LoadSamples()
#37800,25200
x_train, x_test, y_train, y_test = splitTrainTestSet(Samples, Label,10, 0.8)#取得是第15次生成的数据 而且random_state=0表示每次取得15次数据不变 
#法一：
#x_train_oversample, y_train_oversample = oversampleWeakClasses(x_train, y_train)#59216 对整个训练集都进行重采样，校准
# save Preprocessed Data to file
#savePreprocessedData('predata/yancheng_patch_noaug_oversamples/noaug_noover_sameneighbor', x_train, x_test, y_train, y_test, testRatio = 0.4)
#savePreprocessedData('predata/poyang_patch_noaug/noaug_noover_sameneighbor', x_train, x_test, y_train, y_test, testRatio = 0.4)
#savePreprocessedData('predata/manaus_patch_noaug/noaug_noover_sameneighbor_gaosi_saltnoise', x_train, x_test, y_train, y_test, testRatio = 0.4)
savePreprocessedData('D:/Poyang/splitdata/patch/patch_noaug/noaug_noover_sameneighbor/alldata_split', x_train, x_test, y_train, y_test, testRatio = 0.8)

#法二:
#x_all = np.append(x_train, x_test, axis = 0)#6300
#y_all = np.append(y_train, y_test, axis = 0)#
#x_all, y_all = oversampleWeakClasses(x_all, y_all) #98692
#xall_train, xall_test, yall_train, yall_test = splitTrainTestSet(x_all, y_all,10, 0.4)#train: 59215   test:39477
#savePreprocessedData('predata/noaug_oversamples_xall', xall_train, xall_test, yall_train, yall_test, testRatio = 0.4)
#Main()










