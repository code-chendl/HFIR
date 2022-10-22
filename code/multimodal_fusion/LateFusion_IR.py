
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : LateFusion_IR.py
@Time    : 2022/09/10 18:18
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import pickle
import numpy as np
import joblib
from sklearn.metrics import classification_report

#---------------------Load data--------------------------------------------------------#
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

#---------------------Save the file----------------------------------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

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



#load unimodal posterior probabilities calculated by Attention-based BiLSTM, Attention-based VGG19, DeepSentiBank-DNN respectively
t= np.array(load_data(r'all_txt_X_dec_test.pickle'))
v= np.array(load_data(r'all_img_X_dec_test.pickle'))
a = np.array(load_data(r'all_anp_X_dec_test.pickle'))
labels=np.array(load_data(r'all_multi_label_test.pickle'))


pre_label=[]
for i in range(0,len(cor_label_test)):
     #--Perform late fusion with optimal parameters for relevant inputs [Rule1]
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
        #print(decision_label[0])
        pre_label.append(decision_label[0])

    #--Perform late fusion with optimal parameters for irrelevant inputs [Rule2]
    if cor_label_test[i]==1:
        t_elem=[]
        t_elem.append(t[i])
        v_elem=[]
        v_elem.append(v[i])
        a_elem=[]
        a_elem.append(a[i])
        #---------hyper-parameter settings--------------------------
        t_ = np.power(np.array(t_elem), 0.61).astype(np.float32)
        v_ = np.power(np.array(v_elem), 0.35).astype(np.float32)
        a_ = np.power(np.array(a_elem), 0.04).astype(np.float32)
        #------------------------------------------
        decision_f = t_ * v_ * a_ 
        decision_label= decision_f.argmax(axis=1)
        #print(decision_label)
        #print(decision_label[0])
        pre_label.append(decision_label[0])

print(classification_report(labels.argmax(axis=1), pre_label, digits= 5))