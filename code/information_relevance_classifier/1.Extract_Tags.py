from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception 
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import os
import xlwt
import codecs

#------------获取文件列表--------------# 
def fun(sourcePath):
    filelist = []    
    for root, dirs, files in os.walk(sourcePath):
        for fn in files:
            eachpath = str(root+'\\'+fn)
            filelist.append(eachpath)
    return filelist 

#-----------按图片编号加载数据集-------
def order_img(filelist,path):
    ordered_pathlist=[]
    ordered_list=[]
    for ech in filelist:#扫描文档集合
         num=str(ech)[70:]#读取文件名
         print(num)
         num_image=num[:-4]#去掉文件名中的".jpg"
         ordered_list.append(num_image)
    ordered_list.sort()
    for item in ordered_list:
        ordered_path=os.path.join(path,str(item)+'.jpg')
        ordered_pathlist.append(ordered_path)
    return ordered_pathlist

#-------载入图片并预处理-----------
def dealImg(model,path):
    #------部分参数设置--------------
    if model in ("vgg16","vgg19","resnet"): 
        inputShape = (224, 224)
        preprocess = imagenet_utils.preprocess_input #预处理函数为keras.preprocess_input
    if model in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = preprocess_input #预处理函数为separate pre-processing函数
    #-------载入图片并预处理-----------
    print("[INFO] loading and pre-processing image...")
    filelist=fun(path) # 获取文件列表
    ordered_pathlist=order_img(filelist,path)
    image_arr=[]
    for ech in ordered_pathlist:#扫描文档集合
        image = load_img(ech, target_size=inputShape)
        image = img_to_array(image)#转化为矩阵(inputShape[0], inputShape[1], 3)
        image = np.expand_dims(image, axis=0)#添加一个额外的维度，批量训练/分类图片，矩阵(1, inputShape[0], inputShape[1], 3)
        image = preprocess(image)#使用合适的预处理函数执行mean subtraction/scaling
        image_arr.append(image)
    return image_arr

def classifyTag(model,image_arr):
#-------定义五种图像识别模型--------------
    MODELS = {
        "vgg16": VGG16,
        "vgg19": VGG19,
        "inception": InceptionV3,
        "xception": Xception, # Xception只兼容TensorFlow后端
        "resnet": ResNet50
       }
#-------从磁盘载入ImageNet预训练网络的权重，并实例化模型-----
    print("[INFO] loading {}...".format(model))
    Network = MODELS[model]
    model = Network(weights="imagenet")
#-------输入图片特征并获得分类tag-------
    print("[INFO] classifying image with '{}'...".format(model))
    tagList=[]
    for img in image_arr:
        preds = model.predict(img) #从CNN返回预测值
        P =imagenet_utils.decode_predictions(preds) #将预测值解码为易读的键值对：标签、以及该标签的概率
        tag=[]
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            tag.append(label)#返回最可能的5个预测值P[0]
        tagList.append(tag)
    return tagList


file_path = os.path.abspath(".")+"/resultImage"
print(len(file_path))
image_arr_vgg16_19_resnet= dealImg('vgg16',file_path)
image_arr_inception_xception=dealImg('inception',file_path)
#print(len(image_arr))
tag_vgg16=classifyTag('vgg16',image_arr_vgg16_19_resnet)
tag_vgg19=classifyTag('vgg19',image_arr_vgg16_19_resnet)
tag_resnet=classifyTag('resnet',image_arr_vgg16_19_resnet)
tag_inception=classifyTag('inception',image_arr_inception_xception)
tag_xception=classifyTag('xception',image_arr_inception_xception)

#将结果依次写入excel--------------------
workbook = xlwt.Workbook(encoding="ascii")
worksheet = workbook.add_sheet('Tag')
for i in range(0,len(tag_vgg16)):
    worksheet.write(i+1,0,label =[item+' ' for item in tag_vgg16[i]])
    worksheet.write(i+1,1,label =[item+' ' for item in tag_vgg19[i]])
    worksheet.write(i+1,2,label =[item+' ' for item in tag_resnet[i]])
    worksheet.write(i+1,3,label =[item+' ' for item in tag_inception[i]])
    worksheet.write(i+1,4,label =[item+' ' for item in tag_xception[i]])
workbook.save('Tag_result.xls')




