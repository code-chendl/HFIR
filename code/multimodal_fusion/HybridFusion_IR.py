#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : HybridFusion_IR.py
@Time    : 2022/09/10 18:33
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import pickle
import time
import numpy as np
from sklearn.metrics import classification_report
from keras.models import load_model
from TensorFusion_Layer import TensorFusion
from attention.layers import AttentionLayer
import joblib

#---------------------Load data--------------------------------------------------------#
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

#---------------------Save the file----------------------------------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)


root="/gpfs/share/home/HFIR/DataSets/MSA-IR/"
train_X_txt_Path=root+"text/all_txt_arr_train.pickle"
train_X_img_Path=root+"img/all_img_arr_train.pickle"
train_X_anp_Path=root+"anp/"+"all_anp_arr_train.pickle"
train_Y_Path=root+"label/all_multi_label_train.pickle"

test_X_txt_Path=root+"text/all_txt_arr_test.pickle"
test_X_img_Path=root+"img/all_img_arr_test.pickle"
test_X_anp_Path=root+"anp/"+"all_anp_arr_test.pickle"
test_Y_Path=root+"label/all_multi_label_test.pickle"

train_X_txt=load_data(train_X_txt_Path)
train_X_img=load_data(train_X_img_Path)
train_X_anp=load_data(train_X_anp_Path)
train_Y=load_data(train_Y_Path)

test_X_txt=load_data(test_X_txt_Path)
test_X_img=load_data(test_X_img_Path)
test_X_anp=load_data(test_X_anp_Path)
test_Y=load_data(test_Y_Path)

#load unimodal posterior probabilities calculated by Attention-based BiLSTM, Attention-based VGG19, DeepSentiBank-DNN respectively
t= np.array(load_data(r'all_txt_X_dec_test.pickle'))
v= np.array(load_data(r'all_img_X_dec_test.pickle'))
a = np.array(load_data(r'all_anp_X_dec_test.pickle'))

txt_arr=test_X_txt
img_arr=test_X_img
# anp_arr=np.array(load_data(r'all_anp_arr_test.pickle'))
labels=test_Y

#Classify the information relevance between visual content and textual description
root2="/gpfs/share/home/HFIR/consis_classification/consis_classification_uni/"
txt_arr_fea=load_data(root2+'all_txt_X_fea_test.pickle')
txt_arr_fea=np.array(txt_arr_fea)
img_arr_fea=load_data(root2+'all_img_X_fea_test.pickle')
img_arr_fea=np.array(img_arr_fea)
tag_arr=load_data(root2+'all_tag_arr_test.pickle')
tag_arr=np.array(tag_arr)
cos_arr=load_data(root2+'all_cos_arr_test.pickle')
cos_arr=np.array(cos_arr)
multi_arr=np.concatenate((txt_arr_fea,img_arr_fea,tag_arr,cos_arr),axis=1) #axis=1 represents concatenating arrays by rows
cor_model= joblib.load("cor_classifier_uni.h5") #load the pre-trained information relevance classifier
cor_label_test = cor_model.predict(multi_arr) #Determine whether the text and image are consistent with each other

#cor_label_test=np.array(load_data(r'all_cor_label_test.pickle')).argmax(axis= 1) #Relevant: cor_label=0, Irrelevant: cor_label=1



#--------------load the pre-trained intermediate fusion-based component-----------------------
IFModelPath="/gpfs/share/home/HFIR/BestModel/"+'INTER_TFN_uncor_256256_74.h5'
IF_model = load_model(IFModelPath,custom_objects={'TensorFusion': TensorFusion,'AttentionLayer': AttentionLayer})

start_test =time.perf_counter()
pre_label=[]
for i in range(0,len(cor_label_test)):
     #--Perform late fusion with optimal parameters for relevant inputs
    if cor_label_test[i]==0:
        t_elem=[]
        t_elem.append(t[i])
        v_elem=[]
        v_elem.append(v[i])
        a_elem=[]
        a_elem.append(a[i])
        #---------hyper-parameter settings--------------------------
        t_ = np.power(np.array(t_elem), 0.63).astype(np.float32)
        v_ = np.power(np.array(v_elem), 0.29).astype(np.float32)
        a_ = np.power(np.array(a_elem), 0.08).astype(np.float32)
        #------------------------------------------
        decision_f = t_ * v_ * a_ 
        decision_label= decision_f.argmax(axis=1)
        #print(decision_label)
        print(decision_label[0])
        pre_label.append(decision_label[0])

    #--Perform intermediate fusion for irrelevant inputs
    if cor_label_test[i]==1:
        txt_elem=[]
        txt_elem.append(txt_arr[i])
        img_elem=[]
        img_elem.append(img_arr[i])
        # anp_elem=[]
        # anp_elem.append(anp_arr[i])
        inter_label = IF_model.predict([txt_elem,img_elem]).argmax(axis=1)
        #print(inter_label)
        #print(inter_label[0])
        pre_label.append(inter_label[0])
end_test =time.perf_counter()
print(classification_report(labels.argmax(axis=1), pre_label, digits= 5))
print('test time: %s Seconds'%(end_test-start_test))
print(pre_label)
save_data(pre_label,"predictions.pickle")
