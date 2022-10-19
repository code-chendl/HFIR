import os 
import pandas as pd
import numpy as np
from keras.preprocessing import image
import pickle
import cv2

def getImageList(file_dir): 
    result={}
    temp=os.listdir(file_dir)
    for x in temp:
        imageID=x.split(".")[0]
        imagePath=file_dir+x
        result[imageID]=imagePath
    return result

def imageProcess(imageFile):
    img=cv2.imread(imageFile)
    return img


basePath="F:\\ASIST2021\\code\\"
imageDataPath=basePath+"image\\resultImage\\"
trainPath=basePath+"train\\"
testPath=basePath+"test\\"

fileName="uncor_id_train_new.xlsx"
filePath=trainPath+fileName
taskFile=pd.read_excel(filePath,encoding='utf8',header=None)

resultName="uncor_img_arr_train_new.pickle"

imagePathDict=getImageList(imageDataPath)
result=[]
for temp in taskFile.iterrows():
    imageID=temp[1][0]
    if imageID<1000:
        imageIDStr= '%04d' % imageID
    else:
        imageIDStr=str(imageID)
    imagePath=imagePathDict[imageIDStr]
    img=imageProcess(imagePath)
    result.append(img)

img_arrs=[cv2.resize(img_arr, (224, 224)) for img_arr in result]
img_arr_result = np.array(img_arrs).astype(np.float32)/ 255.0
print(img_arr_result.shape)

data_output = open(resultName,'wb')
pickle.dump(img_arr_result,data_output)
data_output.close()




