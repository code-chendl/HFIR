
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : Extract_tags.py
@Time    : 2022/09/09 15:05
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import os
import xlwt
import numpy as np
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception 
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


#------------Get the file list--------------# 
def fun(sourcePath):
    filelist = []    
    for root, dirs, files in os.walk(sourcePath):
        for fn in files:
            eachpath = str(root+'\\'+fn)
            filelist.append(eachpath)
    return filelist 

#-----------Load data by image_ID------------#
def order_img(filelist,path):
    ordered_pathlist=[]
    ordered_list=[]
    for ech in filelist:
         num=str(ech)[70:]     #read the Filename
         print(num)
         num_image=num[:-4]    #remove ".jpg"
         ordered_list.append(num_image)
    ordered_list.sort()
    for item in ordered_list:
        ordered_path=os.path.join(path,str(item)+'.jpg')
        ordered_pathlist.append(ordered_path)
    return ordered_pathlist

#-------Loading images and preprocessing-----------
def dealImg(model,path):
    #------(Partial) Parameter Settings--------------
    if model in ("vgg16","vgg19","resnet"): 
        inputShape = (224, 224)
        preprocess = imagenet_utils.preprocess_input #Function: keras.preprocess_input
    if model in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = preprocess_input                #Function: separate pre-processing
    #------------------------------------------------
    print("[INFO] loading and pre-processing image...")
    filelist=fun(path) # get the file list
    ordered_pathlist=order_img(filelist,path)
    image_arr=[]
    for ech in ordered_pathlist:
        image = load_img(ech, target_size=inputShape)
        image = img_to_array(image)                 #convert into a matrix (inputShape[0], inputShape[1], 3)
        image = np.expand_dims(image, axis=0)       #add an extra dimension, (1, inputShape[0], inputShape[1], 3)
        image = preprocess(image)                   #use the appropriate function to perform: mean subtraction/scaling
        image_arr.append(image)
    return image_arr

def classifyTag(model,image_arr):
#-------Define 5 state-of-the-art CNN-based models in visual object recognition---------------
    MODELS = {
        "vgg16": VGG16,
        "vgg19": VGG19,
        "inception": InceptionV3,
        "xception": Xception,     #Note that Xception is only compatible with the TensorFlow backend
        "resnet": ResNet50
       }
#-------Load the weights of pre-trained networks and instantiate models-----
    print("[INFO] loading {}...".format(model))
    Network = MODELS[model]
    model = Network(weights="imagenet")
#-------input: image features; output: tags ----------------------------------
    print("[INFO] classifying image with '{}'...".format(model))
    tagList=[]
    for img in image_arr:
        preds = model.predict(img)    #output probability scores
        P =imagenet_utils.decode_predictions(preds)  #decode the predicted results into readable key-value pairs
        tag=[]
        for (i, (imagenetID, label, prob)) in enumerate(P[0]):
            tag.append(label)         #extract the top 5 tags
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

#-----------write the result to excel--------------------
workbook = xlwt.Workbook(encoding="ascii")
worksheet = workbook.add_sheet('Tag')
for i in range(0,len(tag_vgg16)):
    worksheet.write(i+1,0,label =[item+' ' for item in tag_vgg16[i]])
    worksheet.write(i+1,1,label =[item+' ' for item in tag_vgg19[i]])
    worksheet.write(i+1,2,label =[item+' ' for item in tag_resnet[i]])
    worksheet.write(i+1,3,label =[item+' ' for item in tag_inception[i]])
    worksheet.write(i+1,4,label =[item+' ' for item in tag_xception[i]])
workbook.save('Tag_result.xls')




