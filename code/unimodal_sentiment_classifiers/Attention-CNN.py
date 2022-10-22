#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : Attention-CNN.py
@Time    : 2022/09/09 16:55
@Author  : Danlei Chen, Wang Su
@Version : 1.0
'''

import pickle
import time
import pandas as pd
import keras
from attention.layers import AttentionLayer
from keras.applications import VGG19
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Reshape
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.utils import CustomObjectScope
from keras import regularizers
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


#------------Load visual features--------------------------------------------#
def load_img(path):
    with open(path, 'rb')as f:
        imgs = pickle.load(f)
    return imgs

#--------------Load the target variable--------------------------------------#
def get_labels(path):
    labels = pd.read_excel(path).label
    labels = LabelBinarizer().fit_transform(labels)
    return labels
#----------------Save the file-----------------------------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

#-----------------Construct iterators-----------------------------------------#
def generator(trainX, trainY):
    while 1:
        for i in range(0, len(trainX), 32):
            X_batch = trainX[i:i + 32]
            Y_batch = trainY[i:i + 32]
            yield (X_batch, Y_batch)

#------------Record accuracy and loss over epochs, and plot the accuracy and loss curves------------#
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):  #invoke at the beginning of training process
        self.losses = {'batch': [], 'epoch': []} 
        self.accuracy = {'batch': [], 'epoch': []} 
        self.val_loss = {'batch': [], 'epoch': []} 
        self.val_acc = {'batch': [], 'epoch': []} 

    def on_batch_end(self, batch, logs={}): #invoke at end of every batch
        self.losses['batch'].append(logs.get('loss'))  #training loss
        self.accuracy['batch'].append(logs.get('acc'))  #training accuracy
        self.val_loss['batch'].append(logs.get('val_loss'))  #validation loss
        self.val_acc['batch'].append(logs.get('val_acc'))  #validation accuracy

    def on_epoch_end(self, batch, logs={}): #invoke at end of every epoch
        self.losses['epoch'].append(logs.get('loss')) #training accuracy
        self.accuracy['epoch'].append(logs.get('acc')) #training accuracy
        self.val_loss['epoch'].append(logs.get('val_loss')) #validation loss
        self.val_acc['epoch'].append(logs.get('val_acc')) #validation accuracy

    def loss_plot_acc(self, loss_type): #Visualize accuracy over epochs
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.val_acc[loss_type], 'g', label='val acc')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.legend(loc="lower right")
        plt.savefig(r'V1_acc.png')
        plt.show()
        
    def loss_plot_loss(self, loss_type): #Visualize loss over epochs
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'b', label='train loss')
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig(r'V1_loss.png')
        plt.show()
       

# -------------------------Fine-tune VGG19---------------------------#
def build_model(trainX,testX,trainY,testY,classNames,taskSavePath):
    #Load pre-trained VGG19 (without fully connected layers after the last convolutional layer)
    baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))) 
    headModel = baseModel.get_layer('block5_conv4').output
    # Reshape the input (14,14,512) to form the (196,512) vector
    headModel = Reshape((196, 512))(headModel)
    headModel = AttentionLayer(name='attention')(headModel)
    # Add 3 FC layers, and the last one is used for classification;
    headModel = Dense(512,W_regularizer=regularizers.l2(0.01), activation="relu")(headModel)
    headModel = Dense(256, W_regularizer=regularizers.l2(0.01), activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # Freeze Conv layers and FC layers of VGG19
    for layer in baseModel.layers:
        layer.trainable = False

    # Recompile the fine-tuned model---------------
    print('-----Recompile the initialized model-----')
    opt = SGD(lr=0.001, momentum= 0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    model.summary() #Print the structure and parameters of the network
    print('-----The fine-tuning model is built-----')

    # The newly added FC layers are initialized and trained
    history = LossHistory()

    csv = CSVLogger(taskSavePath+'log_csv_img_VGG19_V1.csv')

    best_checkpoint = taskSavePath+"VGG19_img_V1.h5"
    modelcheckpoint = ModelCheckpoint(best_checkpoint, monitor= 'val_acc', save_best_only= True, 
                                  mode= 'max', verbose= 1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10,mode='max')
    callbacks_list= [early_stopping,modelcheckpoint,history, csv]

    start_train =time.perf_counter() #The start time of the training process
    model.fit_generator(generator(trainX, trainY),
                        validation_data=(testX, testY), epochs=100,
                        steps_per_epoch=len(trainX) // 32, verbose=1,
                        callbacks=callbacks_list)
    end_train =time.perf_counter()
    score = model.evaluate(testX, testY, batch_size=32)

    history.loss_plot_acc('epoch')
    history.loss_plot_loss('epoch')
    with CustomObjectScope({'AttentionLayer': AttentionLayer}):
        model_img = load_model(best_checkpoint)

    start_test =time.perf_counter()
    predictions = model_img.predict(testX, batch_size=32)
    end_test =time.perf_counter()

    save_data(predictions,taskSavePath+"predictions.pickle") #Save predictions
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=classNames, digits= 5))
    

    print("print the computational efficiency: \n")
    print(start_train)
    print("\n")
    print(end_train)
    print("\n")
    print(start_test)
    print("\n")
    print(end_test)
    print("\n")
    print('Running time: %s Seconds'%(end_train-start_train))
    print('test time: %s Seconds'%(end_test-start_test))

if __name__ == '__main__':
    #load the training set and testing set
    basePath="/gpfs/share/home/HFIR/DataSets/MSA-IR/"
    taskSavePath="Our_SP1/" #taskSavePath is the location where output files are saved during the training process
    trainX=load_img(basePath+"img/"+'all_img_arr_train.pickle')
    trainY = load_img(basePath+"label/"+'all_img_label_train.pickle')
    testX=load_img(basePath+"img/"+'all_img_arr_test.pickle')
    testY = load_img(basePath+"label/"+'all_img_label_test.pickle')

    classNames = ['positive', 'neutral', 'negative']
    #Conduct the fine-tuning and evaluation
    build_model(trainX,testX,trainY,testY,classNames,taskSavePath)
