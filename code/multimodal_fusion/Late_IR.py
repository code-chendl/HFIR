import pickle
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from decimal import Decimal

#读取pickle-----------
def load_data(path):
    with open(path, 'rb')as f:
        data = pickle.load(f)
    return data

#存储-------------------------
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)


cor_label_test=np.array(load_data(r'all_cor_label_test.pickle')).argmax(axis= 1)#cor_label=0为图文相关，1为图文无关

#加载预训练BiLSTM、vgg19、anp-dnn单模态情感分类模型的输出结果，作为决策层融合的输入
t= np.array(load_data(r'all_txt_X_dec_test.pickle'))
v= np.array(load_data(r'all_img_X_dec_test.pickle'))
a = np.array(load_data(r'all_anp_X_dec_test.pickle'))
labels=np.array(load_data(r'all_multi_label_test.pickle'))


pre_label=[]
for i in range(0,len(cor_label_test)):
     #--若图文相关则采用预训练的决策层融合最优权重设置Rule1
    if cor_label_test[i]==0:
        t_elem=[]
        t_elem.append(t[i])
        v_elem=[]
        v_elem.append(v[i])
        a_elem=[]
        a_elem.append(a[i])
        #---------待设置--------------------------
        t_ = np.power(np.array(t_elem), 0.63).astype(np.float32)
        v_ = np.power(np.array(v_elem), 0.29).astype(np.float32)
        a_ = np.power(np.array(a_elem), 0.08).astype(np.float32)
        #------------------------------------------
        decision_f = t_ * v_ * a_ 
        decision_label= decision_f.argmax(axis=1)
        #print(decision_label)
        #print(decision_label[0])
        pre_label.append(decision_label[0])

    #--若图文无关则采用预训练的决策层融合最优权重设置Rule2
    if cor_label_test[i]==1:
        t_elem=[]
        t_elem.append(t[i])
        v_elem=[]
        v_elem.append(v[i])
        a_elem=[]
        a_elem.append(a[i])
        #---------待设置--------------------------
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

#-----绘制混淆矩阵------------------------------------------
cm=confusion_matrix(labels.argmax(axis=1), pre_label)#混淆矩阵计算
p_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]#转化为precision
myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
plt.imshow(p_cm, cmap=plt.cm.Blues)
indices = range(len(p_cm))
plt.xticks(indices, ['正面', '中性', '负面'],fontproperties=myfont,fontsize=15)#先设置字体再设置字号，反过来则没反应
plt.yticks(indices, ['正面', '中性', '负面'],fontproperties=myfont,fontsize=15)   
plt.colorbar(fraction=0.05, pad=0.05)
plt.xlabel('预测值',fontproperties=myfont,fontsize=15)
plt.ylabel('真实值',fontproperties=myfont,fontsize=15)
#plt.title('混淆矩阵',fontproperties=myfont,fontsize=15)
# 显示数据
for first_index in range(len(p_cm)):    #第几列
    for second_index in range(len(p_cm[first_index])):    #第几行
        num=Decimal(p_cm[second_index][first_index]).quantize(Decimal('0.00'))
        plt.text(first_index,second_index,num,fontproperties='Times New Roman',fontsize=15)
plt.savefig(r'p_cm_late.png')
# 显示
plt.show()
