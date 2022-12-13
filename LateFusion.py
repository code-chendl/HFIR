#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : LateFusion.py
@Time    : 2022/09/11 20:11
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import pickle
import numpy as np
import xlwt
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score,accuracy_score
from keras.models import Model, load_model
from attention.layers import AttentionLayer
from sklearn.model_selection import train_test_split


#-----------------Load data-----------------------#
def load_f(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

#----------Save the file--------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)


basePath="/gpfs/share/home/HFIR/DataSets/twitter1517/"
modelBasePath="/gpfs/share/home/HFIR/twitter1517_Over_BestModel_20221016/"

X_text=load_f(basePath+"text_over/"+"twitter1517_cor_text_arr.pickle")
X_img=load_f(basePath+"img_over/"+"cor_image_arr.pickle")
X_anp=load_f(basePath+"anp_over/"+"twitter1517_OS_cor_arr.pickle")
Y=load_f(basePath+"label_over/"+'cor_multi_label.pickle')
train_X_txt, test_X_txt, train_Y, test_Y = train_test_split(X_text, Y, test_size=0.2, random_state=1) 
train_X_img, test_X_img, train_Y, test_Y = train_test_split(X_img, Y, test_size=0.2, random_state=1) 
train_X_anp, test_X_anp, train_Y, test_Y = train_test_split(X_anp, Y, test_size=0.2, random_state=1) 

img_arr=train_X_img
txt_arr=train_X_txt
anp_arr=train_X_anp
labels=train_Y


# -------Extract visual posterior probabilities calculated by Attention-based VGG19-------------
imgmodel = load_model(modelBasePath+'VGG19_img_V1.h5',custom_objects={'AttentionLayer': AttentionLayer})
#print(imgmodel.summary())
layer_name = 'dense_3'
conca_layer =  Model(inputs= imgmodel.input, 
                    outputs= imgmodel.get_layer(layer_name).output)
img_X = conca_layer.predict(np.array(img_arr), batch_size= 32) 
#print(img_X.shape),(4800,3)
save_data(img_X, r'cor_img_X_dec_train.pickle')  


#----Extract textual posterior probabilities calculated by Attention-based BiLSTM------------
txtmodel = load_model(modelBasePath+'BiLSTM_SelfAttention_Text.h5',custom_objects={'AttentionLayer': AttentionLayer})
#print(txtmodel.summary())
layer_name = 'dense_1'
conca_layer =  Model(inputs= txtmodel.input, 
                    outputs= txtmodel.get_layer(layer_name).output)
txt_X = conca_layer.predict(np.array(txt_arr), batch_size= 32)
#print(txt_X.shape),(4800,3)
save_data(txt_X, r'cor_txt_X_dec_train.pickle')


#----Extract (mid-level) visual posterior probabilities calculated by DeepSentiBank-DNN------------
# anpmodel= joblib.load(r'best_anp_svm.h5')
# anp_X = anpmodel.predict(np.array(anp_arr))
#anpmodel = load_model(r'all_anp_dnn_l2_best.h5')
#layer_name = 'dense_4'
anpmodel = load_model(modelBasePath+'ANPDNN.h5')
layer_name = 'dense_3'
conca_layer =  Model(inputs= anpmodel.input, 
                    outputs= anpmodel.get_layer(layer_name).output)
anp_X = conca_layer.predict(np.array(anp_arr), batch_size= 32)
#print(txt_X.shape),(4800,3)
print(anp_X[1])
save_data(anp_X, r'cor_anp_X_dec_train.pickle')




#--------Fine-grained decision-level fusion, using the weighted product-rule--------------
t = load_f(r'cor_txt_X_dec_train.pickle')
v = load_f(r'cor_img_X_dec_train.pickle')
a = load_f(r'cor_anp_X_dec_train.pickle')


result=np.zeros((101,101,4)) #(p,r,f1,acc)
for i in range(0, 101, 1):
    for j in range(0,101-i,1):
        t_ = np.power(np.array(t), i*0.01).astype(np.float32)
        v_ = np.power(np.array(v), j*0.01).astype(np.float32)
        a_ = np.power(np.array(a), (100-i-j)*0.01).astype(np.float32)
        decision_f = t_ * v_ * a_ 
        decision_label = decision_f.argmax(axis = 1)
        precision = precision_score(labels.argmax(axis= 1), decision_label, average= 'macro') 
        recall = recall_score(labels.argmax(axis= 1), decision_label, average= 'macro') 
        f_1 = f1_score(labels.argmax(axis= 1), decision_label, average= 'macro') 
        accuracy=accuracy_score(labels.argmax(axis= 1), decision_label)

        print(classification_report(labels.argmax(axis= 1), decision_label, digits= 5))
        result[i][j][0]=precision
        result[i][j][1]=recall
        result[i][j][2]=f_1
        result[i][j][3]=accuracy


#-----------write the result to excel--------------------
workbook = xlwt.Workbook(encoding="ascii")
worksheet_P = workbook.add_sheet('lateF_result_P')
worksheet_R = workbook.add_sheet('lateF_result_R')
worksheet_F = workbook.add_sheet('lateF_result_F')
worksheet_A = workbook.add_sheet('lateF_result_A')
for i in range(0, 101, 1):
    for j in range(0,101-i,1):
        worksheet_P.write(i,j,label =result[i][j][0])
        worksheet_R.write(i,j,label =result[i][j][1])
        worksheet_F.write(i,j,label =result[i][j][2])
        worksheet_A.write(i,j,label =result[i][j][3])
workbook.save('cor_lateF_train_product_weightedmini_acc.xls')




