# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:28:55 2019

@author: ZHOUFENG
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
from sklearn.decomposition import PCA
import gdal
from sklearn import preprocessing
import mahotas as mh
import os
import random

#读取Tiff文件
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
        return
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_bands = dataset.RasterCount #波段数
    #im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
    mat1 = np.zeros(((im_height,im_width,im_bands)))
    for i in range(0,im_bands):
         # 获取影像的第i+1个波段
        band_i = dataset.GetRasterBand(i + 1)
        #mat1[:,:,i] = im_data[i,0:im_height,0:im_width]
        mat1[:,:,i]=band_i.ReadAsArray(0,0,im_width,im_height)
    return mat1

#添加高斯白噪声
def GaussianNoise(X,means,sigma,percetage):
    NoiseImg = X.copy()
    NoiseNum = int(percetage * X.shape[0] * X.shape[1] * X.shape[2])
    for i in range(NoiseNum):
        randX = random.randint(0,X.shape[0] - 1)
        randY = random.randint(0,X.shape[1] - 1)
        randZ = random.randint(0,X.shape[2] - 1)
        NoiseImg[randX,randY,randZ] = NoiseImg[randX,randY,randZ] + random.gauss(means,sigma)
        if NoiseImg[randX, randY,randZ] < 0:
            NoiseImg[randX, randY,randZ] = 0
        if NoiseImg[randX, randY,randZ] > 255:
            NoiseImg[randX, randY,randZ] = 255
    return NoiseImg

#添加椒盐噪声
def addsalt_pepper(img, SNR):
    img = np.transpose(img,(2,0,1))
    img_ = img.copy()
    c,h, w  = img_.shape
    #print(h,w,c)
    mask = np.random.choice((0, 1, 2), size = (1,h, w), p = [SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    #print(mask)
    mask = np.repeat(mask, c, axis = 0)     # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 255    # 盐噪声
    img_[mask == 2] = 0      # 椒噪声
    return img_

#Z-score(标准化,均值为0方差为1)
def standartizeData(X):
    newX = np.reshape(X, (-1, X.shape[2]))#63000*155
    scaler = preprocessing.StandardScaler().fit(newX)  #63000*155
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    return newX

#  用PCA对数据进行预处理，返回降维后的张量
def applyPCA(X, numComponents = 30):
    #将数据变成了19701*158的二维数组
    newX = np.reshape(X, (-1, X.shape[2]))#先把X展开为一个行向量，现在行数未知，列数为158，那自动计算出行数为19701
   #取主成分为30，对158的波段数进行降维
    pca = PCA(n_components = numComponents, whiten = True)
    newX = pca.fit_transform(newX)#19701 * 30
    #将降维后的矩阵又还原为一个199*99*30的张量
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

# padding补0操作   pad zeros to dataset，返回201*101*30
def padWithZeros(X, margin = 1):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))#201*101*30
    x_offset = margin 
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X #newx中的1:200行，1:100列为原始数据；
    return newX #201*101*30

def padWithSame(mat1):
    mat_new = np.zeros(((mat1.shape[0] + 2,mat1.shape[1] + 2,mat1.shape[2])))
    for i in range(0,mat1.shape[0]):
        for j in range(0,mat1.shape[1]):
            #print(mat1[i,j])
            mat_new[i + 1,j + 1] = mat1[i,j]
            if i == 0 and j == 0:
                mat_new[i,j] = mat1[i,j]#左上角那块
            if i == 0 and j == mat1.shape[1] - 1:
                mat_new[i,j + 2] = mat1[i,j]#右上角那块
            if i == 0:
                mat_new[i,j + 1] = mat1[i,j]#第一行中间的剩余部分
            if i == mat1.shape[0] - 1 and j == 0:
                mat_new[i + 2,j] = mat1[i,j]#左下角那块
            if i == mat1.shape[0] - 1 and mat1.shape[1] - 1:
                mat_new[i + 2,j + 2] = mat1[i,j]#右下角那块
            if i == mat1.shape[0] - 1:
                mat_new[i + 2,j + 1] = mat1[i,j]#最后一行中间剩余部分
            if j == 0:
                mat_new[i + 1,j] = mat1[i,j]#最左边中间剩余
            if j == mat1.shape[1] - 1:
                mat_new[i + 1,j + 2] = mat1[i,j]#最右边中间剩余
                #print(mat_new[i+1,j+1],mat1[i,j])
   # print('after addneighbors:')
    #print(mat_new)#(452, 142, 30)
    return mat_new

#  为数据集创建patch        create Patches for dataset
def createPatches(X, windowSize = 3):
    margin = int((windowSize - 1) / 2) #1
    #zeroPaddedX = padWithZeros(X, margin = margin)#在原数据外面补1圈0的新数据 201*101*30
    samePaddedX = padWithSame(X) #在原数据外面补相同数据 201*101*30
    #print(samePaddedX.shape)
    # 对数据进行分割     split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize,  windowSize, X.shape[2]))#19701,3,3,30
    #print(patchesData)
    #patchesLabels = np.zeros((X.shape[0] * X.shape[1]))#19701*1
    patchIndex = 0
    
    #循环给每个像素19701个 构建一个3*3窗口的patch，patch中心即为每个像素点
    for r in range(margin, samePaddedX .shape[0] - margin):#1：200行
        for c in range(margin, samePaddedX .shape[1] - margin):#1:100列
            patch = samePaddedX [r - margin:r + margin + 1, c - margin:c + margin + 1]   #以(r,c)位置为中心的patch，也就是以每个像素点为中心的patch
            patchesData[patchIndex, :, :, :] = patch
            #patchesLabels[patchIndex] = y[r-margin, c-margin] #(199,99)
            patchIndex = patchIndex + 1
    return patchesData

#保存预处理数据文件      save Preprocessed Data to file
def SavePatch(path, T1Patches, T2Patches):
    data_path = os.path.join(os.getcwd(), path)
    with open(os.path.join(data_path, "T1patches_sameneighbor") + ".npy", 'bw') as outfile:
        np.save(outfile, T1Patches)
    with open(os.path.join(data_path, "T2patches_sameneighbor") + ".npy", 'bw') as outfile:
        np.save(outfile, T2Patches)     

#调取tiff文件
#filename1 = 'F:\\b705\\ZHOUFENG\\change\\change\\Manaus\\T1.tif'
#filename2 = 'F:\\b705\\ZHOUFENG\\change\\change\\Manaus\\T2.tif'
#filename3 = 'F:\\b705\\ZHOUFENG\\change\\change\\Manaus\\changeonly_groundtruth.tif'
#filename1 = 'F:\\b705\\ZHOUFENG\\change\\change_detection\\yancheng_allfile\\Yancheng\\HY2006tif.tif'
#filename2 = 'F:\\b705\\ZHOUFENG\\change\\change_detection\\yancheng_allfile\\Yancheng\\HY2007tif.tif'
#filename3 = 'F:\\b705\\ZHOUFENG\\change\\change_detection\\yancheng_allfile\\Yancheng\\changeonly_groundtruth.tif'
#filename1 = 'F:\\b705\\ZHOUFENG\\change\\change_detection\\poyang_allfile\\Poyang\\HSI2010roitif.tif'
#filename2 = 'F:\\b705\\ZHOUFENG\\change\\change_detection\\poyang_allfile\\Poyang\\HSI2011roitif.tif'
#filename3 = 'F:\\b705\\ZHOUFENG\\change\\change_detection\\poyang_allfile\\Poyang\\changeonly_groundtruth.tif'

#读取data文件
#mat1 = readTif(filename1)
#mat2 = readTif(filename2)
#mat3 = readTif(filename3)#numpy.ndarray        
      
#读取mat文件
#mat1 = io.loadmat('F:\\b705\\ZHOUFENG\\change\\change\\River\\river_before.mat')['river_before']
#mat2 = io.loadmat('F:\\b705\\ZHOUFENG\\change\\change\\River\\river_after.mat')['river_after']
#mat3 = io.loadmat('F:\\b705\\ZHOUFENG\\change\\change\\River\\groundtruth.mat')['groundtruth']
    
mat11 = io.loadmat('F:\\b705\\ZHOUFENG\\change\\change_detection\\GAUSSIAN_001_SPARSEGAUSSIAN_05_SPARSEIMPULSE_PIXELS0005_SPARSELINES_8_BANDS20\\SUSTRACTION_HSI_T1_T2_index_0.mat')
mat1 = mat11['T1']

#mat22 = io.loadmat('F:\\b705\\ZHOUFENG\\change\\change_detection\\GAUSSIAN_001_SPARSEGAUSSIAN_05_SPARSEIMPULSE_PIXELS0005_SPARSELINES_8_BANDS20\\SUSTRACTION_HSI_T1_T2_index_0.mat')
mat2 = mat11['T2']
filename3 = 'F:\\b705\\ZHOUFENG\\change\\change_detection\\GAUSSIAN_001_SPARSEGAUSSIAN_05_SPARSEIMPULSE_PIXELS0005_SPARSELINES_8_BANDS20\\changeonly_groundtruth.tif'
mat3 =  readTif(filename3)
mat3=mat3[:,:,0]
#给数据添加高斯白噪声
#img1 = GaussianNoise(mat1,2,6,0.5)
#img2 = GaussianNoise(mat2,2,6,0.5)
#添加椒盐噪声
#img11 = addsalt_pepper(img1,0.7)
#mat1 = np.transpose(img11,(1,2,0))
#img12 = addsalt_pepper(img2,0.7)
#mat2 = np.transpose(img12,(1,2,0))

#Z-score标准化
mat1_standard = standartizeData(mat1)#(199, 99, 158) #(450,140,155) #(200,200,106) #(463,241,198)
mat2_standard = standartizeData(mat2)#(199, 99, 158) #(450,140,155) #(200,200,106) #(463,241,198)

#用pca降维到30
numComponents = 30
T1, pca1 = applyPCA(mat1_standard, numComponents = numComponents)#对原数据进行pca降维，降到前30个主成分 #450*140*30
T2, pca2 = applyPCA(mat2_standard, numComponents = numComponents)#对原数据进行pca降维，降到前30个主成分
#print(T1)

windowSize = 3
#生成patch文件
T1Patches = createPatches(T1,windowSize = windowSize)#(19701,3,3,30) #(63000,3,3,30) #(40000,3,3,30) #(111583,3,3,30)
#print(T1Patches[3,:,:,:])
T2Patches = createPatches(T2,windowSize = windowSize)#(19701,3,3,30) ##(63000,3,3,30) #(40000,3,3,30) #(111583,3,3,30)

#保存patchdata文件
#SavePatch('manaus_predata/patchdata_noise/gaosi_saltnoise', T1Patches, T2Patches)
#SavePatch('yancheng_allfile/yancheng_predata/matsamples/patchdata', T1Patches, T2Patches)
#SavePatch('poyang_predata/patchdata', T1Patches, T2Patches)
#SavePatch('river_predata/patchdata', T1Patches, T2Patches)
#SavePatch('paviau_allfile/paviau_predata/matsamples/patchdata', T1Patches, T2Patches)
#保存nopatch文件
#SavePatch('poyang_allfile/poyang_predata/matsamples/nopatchdata', T1, T2)


#加载npy文件并转为mat格式
##加载zerosneighbor
#mat1 = np.load('manaus_predata/patchdata/T1patches.npy')
#mat2 = np.load('manaus_predata/patchdata/T2patches.npy')
#io.savemat('manaus_predata/patchdata/T1patches.mat', {'T1': mat1})
#io.savemat('manaus_predata/patchdata/T2patches.mat', {'T2': mat2})

##加载sameneighbor
mat1 = np.load('paviau_allfile/paviau_predata/matsamples/patchdata/T1patches_sameneighbor.npy')
mat2 = np.load('paviau_allfile/paviau_predata/matsamples/patchdata/T2patches_sameneighbor.npy')
#io.savemat('paviau_allfile/paviau_predata/matsamples/patchdata/T1patches_sameneighbor.mat', {'T1': mat1})
#io.savemat('paviau_allfile/paviau_predata/matsamples/patchdata/T2patches_sameneighbor.mat', {'T2': mat2})

#保存真实变化标签
#真实标签true
#true = mh.colors.rgb2gray(mat3)
true = mat3
for i in range(0,true.shape[0]):
   for j in range(0,true.shape[1]):
        if true[i][j] == 255:
            true[i][j] = 1
        else:
            true[i][j] = 0 
           

plt.subplot(1,2,2)
plt.title('TrueLab image')
plt.imshow(true,cmap = 'gray')
plt.show()

[u2,v2] = true.shape
t2 = u2*v2
y2 = true.reshape(t2,1)
y2 = y2.flatten()#展平为一维矩阵

#mat_path1 = 'F:\\b705\\ZHOUFENG\\change\\change\\matdata\\manaus\\pca_kmeanslabel.mat'
mat_path2 = 'F:\\b705\\ZHOUFENG\\change\\change_detection\\paviau_allfile\\paviau_predata\\matsamples\\patchdata\\truelabel.mat'
#io.savemat(mat_path2, {'y2': y2})
