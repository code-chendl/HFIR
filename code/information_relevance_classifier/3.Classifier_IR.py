#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : Classifier_IR.py
@Time    : 2022/09/11 20:00
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import xlrd
import pickle
import numpy as np
from sklearn.metrics import classification_report, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import joblib


#--------Load data--------------------#
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

#-------Read excel--------------------#
def readXls(path):
    xl=xlrd.open_workbook(path)
    sheet=xl.sheets()[0]
    data=[]
    for i in range(0,sheet.ncols): # ncols represents: read by column
        data.append(list(sheet.col_values(i)))
    return data

#-------save results in the form of pickles-----------
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

test_Y=load_data(r"all_cor_label_test.pickle")
train_Y=load_data(r"all_cor_label_train.pickle")

#----------------Compute the image-text similarity---------------
#------load training set-------
excel_train=readXls(r'cos_txtTag_uni_train.xlsx')
cos_train=excel_train[0]   #Read the results of cosine similarity approach
cos_arr_train = np.zeros((len(cos_train),1)).astype(np.float32)
for i in range(0,len(cos_train)):
    cos_arr_train[i,:]=cos_train[i]
save_data(cos_arr_train,r'all_cos_arr_train.pickle')
print(cos_arr_train.shape)

#--------load testing set-------
excel_test=readXls(r'cos_txtTag_uni_test.xlsx')
cos_test=excel_test[0]   #Read the results of cosine similarity approach
cos_arr_test = np.zeros((len(cos_test),1)).astype(np.float32)
for i in range(0,len(cos_test)):
    cos_arr_test[i,:]=cos_test[i]
save_data(cos_arr_test,r'all_cos_arr_test.pickle')
print(cos_arr_test.shape)

#--Read the attended textual feature, the attended visual feature, the combined tag-ANP feature;
tag_arr_train=np.array(load_data(r'all_tag_arr_train.pickle')) 
tag_arr_test=np.array(load_data(r'all_tag_arr_test.pickle')) 
txt_arr_train=np.array(load_data(r'all_txt_X_train.pickle')) 
txt_arr_test=np.array(load_data(r'all_txt_X_test.pickle')) 
img_arr_train=np.array(load_data(r'all_img_X_train.pickle')) 
img_arr_test=np.array(load_data(r'all_img_X_test.pickle')) 

#--------------concatenate multimodal features------------------------
multi_arr_train=np.concatenate((txt_arr_train,img_arr_train,tag_arr_train,cos_arr_train),axis=1) #axis=1 represents concatenating arrays by rows
multi_arr_test=np.concatenate((txt_arr_test,img_arr_test,tag_arr_test,cos_arr_test),axis=1) 
print(multi_arr_train.shape)
print(multi_arr_test.shape)

train_Y = train_Y.argmax(axis= 1)
test_Y = test_Y.argmax(axis= 1)

clf_s= SVC(kernel='rbf', gamma= 'auto')
parameters = {'kernel': ['poly', 'rbf', 'linear'],
'C': [pow(2, -3), pow(2, -2), pow(2, -1), 1, pow(2, 3), pow(2, 2), pow(2, 1)]
}
acc_scorer = make_scorer(accuracy_score)
grid_obj = GridSearchCV(clf_s, parameters, scoring=acc_scorer,cv=4)
grid_obj = grid_obj.fit(multi_arr_train, train_Y)
print("Best: %f using %s" % (grid_obj.best_score_,grid_obj.best_params_)) #best_params_:the optimal parameter Settings; best_score_: the performance of best_estimator
clf = grid_obj.best_estimator_
#clf.fit(multi_arr_train, train_Y)
y_pre = clf.predict(multi_arr_test)
print(classification_report(test_Y, y_pre, digits= 5))
joblib.dump(clf,"cor_classifier.h5")  #save the information relevance classifier

