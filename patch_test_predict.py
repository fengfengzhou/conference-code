# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:52:35 2019

@author: ZHOUFENG
"""

import scipy.io as io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
#import itertools
#import spectral
import matplotlib.pyplot as plt
from noaug_patch_truelabel import LoadSamples
#from skimage import filters
from keras.utils import plot_model
import itertools

# Get the model evaluation report, 
# include classification report, confusion matrix, Test_Loss, Test_accuracy
target_names = [ 'non-changed','changed']
def reports(X_test,y_test):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis = 1)

    classification = classification_report(np.argmax(y_test, axis = 1), y_pred, target_names = target_names)
    confusion = confusion_matrix(np.argmax(y_test, axis = 1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size = 32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100
    kc = cohen_kappa_score(np.argmax(y_test, axis = 1), y_pred)
    
    return classification, confusion, Test_Loss, Test_accuracy, kc

# 导入预处理文件
#alltest_count = 49346#40423#49800#32339 #24254#24648
#X_test = np.load("./predata_aug/augafter2_train_49346/X_augafter100_test_" + "alltest_count" + str(alltest_count) + ".npy")
#y_test = np.load("./predata_aug/augafter2_train_49346/y_augafter100_test_" + "alltest_count" + str(alltest_count) + ".npy")
#未进行数据增强
testRatio = 0.9
X_test = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_noover_sameneighbor/alldata_split/X_stand_pca_test_" + "testRatio" + str(testRatio) + ".npy")
y_test = np.load("./poyang_allfile/poyang_predata/splitdata/patch/patch_noaug/noaug_noover_sameneighbor/alldata_split/y_stand_pca_test_" + "testRatio" + str(testRatio) + ".npy")
#alltest_count = 38290
#X_test = np.load("./poyang_allfile/poyang_predata/splitdata/nopatch/nopatch_noaug/noaug_sameneighbor/twotypes_split/X_noaugsplit_test_" + "alltest_count" + str(alltest_count) + ".npy")
#y_test = np.load("./poyang_allfile/poyang_predata/splitdata/nopatch/nopatch_noaug/noaug_sameneighbor/twotypes_split/y_noaugsplit_test_" + "alltest_count" + str(alltest_count) + ".npy")

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]))
y_test = np_utils.to_categorical(y_test)

#model = load_model('./model/river_patch_model/patch_noaug/noaug_sameneighbor/HSI_model_epochs3_150.h5')
model = load_model('./poyang_allfile/poyang_model/patch_model/patch_noaug/noaug_noover_sameneighbor/testratio0.9/HSI_model_epochs2_200.h5')

# 计算结果，损失，精度，混淆矩阵
classification, confusion, Test_loss, Test_accuracy ,kc = reports(X_test,y_test)
classification = str(classification)
confusion_str = str(confusion)

# show result and save to file
print('Test loss {} (%)'.format(Test_loss))
print('Test accuracy {} (%)'.format(Test_accuracy))
print('Kappa {} (%)'.format(kc))
print("classification result: ")
print('{}'.format(classification))
print("confusion matrix: ")
print('{}'.format(confusion_str))
#file_name = './result/100/report_' + "testRatio_" + str(testRatio) +".txt"
file_name = './poyang_allfile/poyang_result/patch_result/patch_noaug/noaug_noover_sameneighbor/testratio0.9/report_' + "testratio_noaugrmsprop2_" + str(testRatio) +".txt"
#file_name = './result/100_augafter2/report_' + "X_augafter22_test_" + str(alltest_count) +".txt"

with open(file_name, 'w') as x_file:
    x_file.write('Test loss {} (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('Test accuracy {} (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('Kappa {} (%)'.format(kc))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write(" classification result: \n")
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write(" confusion matrix: \n")
    x_file.write('{}'.format(confusion_str))

#画混淆矩阵
def plot_confusion_matrix(cm, classes,normalize = False,title = 'Confusion matrix',cmap = plt.get_cmap("Blues")):   
    Normalized = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    if normalize:
        cm = Normalized
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(Normalized, interpolation = 'nearest', cmap = cmap)
    plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        thresh = cm[i].max() / 2.
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#plt.figure(figsize = (5,5))
#plot_confusion_matrix(confusion, classes = target_names, normalize = False, 
                      #title='Confusion matrix, without normalization')
#plt.savefig("./poyang_allfile/poyang_result/patch_result/patch_noaug/noaug_sameneighbor/testratio0.8/confusion_matrix_without_normalization1.svg")
#plt.show()
plt.figure(figsize = (5,5))
plot_confusion_matrix(confusion, classes = target_names, normalize = True, 
                      title = 'Confusion matrix')
plt.savefig("./poyang_allfile/poyang_result/patch_result/patch_noaug/noaug_noover_sameneighbor/testratio0.9/confusion_matrix_with_normalization2.svg")
plt.show()

Samples_Path  = 'D:\\b705\\ZHOUFENG\\change_detection\\poyang_allfile\\poyang_predata\\matsamples\\patchdata\\'   
#(463,241,198)
Truelabel_mat = Samples_Path + 'truelabel.mat'
TrueLab = io.loadmat(Truelabel_mat).get('y2')
TrueLab = TrueLab.reshape((200,200))
height = TrueLab.shape[0]
width = TrueLab.shape[1]
#注意文件夹！！！
Samples,Label = LoadSamples()
# calculate the predicted image
outputs = np.zeros((height,width))
for i in range (0,height):
    for j in range(0,width):
        NodeMatrix = Samples[i * width + j,:,:,:]
        X_test_NodeMatrix = NodeMatrix.reshape(1,Samples.shape[1],Samples.shape[2],Samples.shape[3]).astype('float32')                                 
        prediction = (model.predict_classes(X_test_NodeMatrix))                         
        outputs[i][j] = prediction
            
#ground_truth = spectral.imshow(classes = TrueLab, figsize = (5, 5))
#predict_image = spectral.imshow(classes = outputs.astype(int), figsize = (5, 5))

plt.figure()
plt.subplot(1,2,1)
plt.title('TrueLab_rmsprop image')
plt.imshow(TrueLab,cmap = 'gray')

plt.subplot(1,2,2)
plt.title('Predict_rmsprop image')
plt.imshow(outputs,cmap = 'gray')
plt.savefig("./poyang_allfile/poyang_result/patch_result/patch_noaug/noaug_noover_sameneighbor/testratio0.9/rmsprop image2.svg")


