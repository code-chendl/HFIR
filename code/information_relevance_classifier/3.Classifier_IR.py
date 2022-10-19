import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import joblib
import xlrd

#读取pickle-----------
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

#-----读取excel--------
def readXls(path):
    xl=xlrd.open_workbook(path)
    sheet=xl.sheets()[0]
    data=[]
    for i in range(0,sheet.ncols): # ncols 表示按列读取
        data.append(list(sheet.col_values(i)))
    return data

#存储pickle-------------
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

test_Y=load_data(r"all_cor_label_test.pickle")
train_Y=load_data(r"all_cor_label_train.pickle")
#----------------提取余弦相似度特征---------------
#------训练集-------
excel_train=readXls(r'cos_txtTag_uni_train.xlsx')
cos_train=excel_train[0]   #读取余弦相似度计算结果
cos_arr_train = np.zeros((len(cos_train),1)).astype(np.float32)
for i in range(0,len(cos_train)):
    cos_arr_train[i,:]=cos[i]
save_data(cos_arr_train,r'all_cos_arr_train.pickle')
print(cos_arr_train.shape)

#--------测试集-------
excel_test=readXls(r'cos_txtTag_uni_test.xlsx')
cos_test=excel_test[0]   #读取余弦相似度计算结果
cos_arr_test = np.zeros((len(cos_test),1)).astype(np.float32)
for i in range(0,len(cos_test)):
    cos_arr_test[i,:]=cos[i]
save_data(cos_arr_test,r'all_cos_arr_test.pickle')
print(cos_arr_test.shape)

#---------------读取txt/img/anp/tag特征----------------
tag_arr_train=np.array(load_data(r'all_tag_arr_train.pickle')) #读取标签特征
tag_arr_test=np.array(load_data(r'all_tag_arr_test.pickle')) 
txt_arr_train=np.array(load_data(r'all_txt_X_train.pickle')) 
txt_arr_test=np.array(load_data(r'all_txt_X_test.pickle')) 
img_arr_train=np.array(load_data(r'all_img_X_train.pickle')) 
img_arr_test=np.array(load_data(r'all_img_X_test.pickle')) 

#--------------特征拼接------------------------
multi_arr_train=np.concatenate((txt_arr_train,img_arr_train,tag_arr_train,cos_arr_train),axis=1) #axis=1表示对应行的数组进行拼接 
multi_arr_test=np.concatenate((txt_arr_test,img_arr_test,tag_arr_test,cos_arr_test),axis=1) #axis=1表示对应行的数组进行拼接
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
print("Best: %f using %s" % (grid_obj.best_score_,grid_obj.best_params_))#best_params_：在保存数据上给出最佳结果的参数设置，best_score_：best_estimator的分数
clf = grid_obj.best_estimator_
#clf.fit(multi_arr_train, train_Y)
y_pre = clf.predict(multi_arr_test)
print(classification_report(test_Y, y_pre, digits= 5))
joblib.dump(clf,"cor_classifier.h5")  #将训练好的一致性分类模型保存

# #----------测试所保存的模型是否有效--------
# model_save= joblib.load("cor_classifier.h5")
# y_pre_save = model_save.predict(multi_arr_test)
# print(classification_report(test_Y, y_pre_save, digits= 5))
