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



#------------读取图片特征变量-------------------------------------------------#
def load_img(path):
    with open(path, 'rb')as f:
        imgs = pickle.load(f)
    return imgs
#-------------读取目标变量----------------------------------------------------#
def get_labels(path):
    labels = pd.read_excel(path).label
    labels = LabelBinarizer().fit_transform(labels)
    return labels
#---------------------保存文件----------------------------------------------------#
def save_data(data, path):
    with open(path, 'wb')as f:
        pickle.dump(data, f)

#-----------------构造迭代器--------------------------------------------------#
def generator(trainX, trainY):
    while 1:
        for i in range(0, len(trainX), 32):
            X_batch = trainX[i:i + 32]
            Y_batch = trainY[i:i + 32]
            yield (X_batch, Y_batch)

#------------记录并可视化训练集、测试集中准确率、损失值随训练轮数的变化------------#
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):  #on_train_begin: 在模型训练开始时被调用
        self.losses = {'batch': [], 'epoch': []} #在训练时，保存一个列表的批量损失值【训练集中】
        self.accuracy = {'batch': [], 'epoch': []} #在训练时，保存一个列表的准确率值【训练集中】
        self.val_loss = {'batch': [], 'epoch': []} #在训练时，保存一个列表的批量损失值【测试集中】
        self.val_acc = {'batch': [], 'epoch': []} #在训练时，保存一个列表的准确率值【测试集中】

    def on_batch_end(self, batch, logs={}): #on_batch_end: 在每批结束时被调用
        self.losses['batch'].append(logs.get('loss'))  #记录训练集损失值
        self.accuracy['batch'].append(logs.get('acc'))  #记录训练集准确率
        self.val_loss['batch'].append(logs.get('val_loss'))  #记录测试集损失值
        self.val_acc['batch'].append(logs.get('val_acc'))  #记录测试集准确率

    def on_epoch_end(self, batch, logs={}): #on_epoch_end: 在每轮结束时被调用
        self.losses['epoch'].append(logs.get('loss')) #记录训练集损失值
        self.accuracy['epoch'].append(logs.get('acc')) #记录训练集准确率
        self.val_loss['epoch'].append(logs.get('val_loss')) #记录测试集损失值
        self.val_acc['epoch'].append(logs.get('val_acc')) #记录测试集准确率

    def loss_plot_acc(self, loss_type): #可视化准确率随训练轮数的变化
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
        
    def loss_plot_loss(self, loss_type): #可视化损失值随训练轮数的变化
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
       

# -------------------------构造基于迁移学习的VGG19模型---------------------------#
def build_model(trainX,testX,trainY,testY,classNames,taskSavePath):
    #加载VGG19模型（不含有最后一层卷积层后的全连接层）
    baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))) 
    headModel = baseModel.get_layer('block5_conv4').output
    #将输入（14，14，512）压平，形成（196,512）向量;
    headModel = Reshape((196, 512))(headModel)
    headModel = AttentionLayer(name='attention')(headModel)
    #加入三层全连层，最后一层全连层用于分类;
    headModel = Dense(512,W_regularizer=regularizers.l2(0.01), activation="relu")(headModel)
    headModel = Dense(256, W_regularizer=regularizers.l2(0.01), activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # 对于VGG19模型中的卷积层和全连接层不进行训练；
    for layer in baseModel.layers:
        layer.trainable = False

    # 对微调后的模型重新编译；
    print('-----重新编译初始化后的模型-----')
    opt = SGD(lr=0.001, momentum= 0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    model.summary() #打印网络结构和参数状况


    # 再次训练已构建的模型，对基准模型的最后一层卷积层和新加入的全连接层重新训练以及初始化；
    print('-----微调模型构建完毕-----')
    history = LossHistory()

    csv = CSVLogger(taskSavePath+'log_csv_img_VGG19_V1.csv')

    best_checkpoint = taskSavePath+"VGG19_img_V1.h5"
    modelcheckpoint = ModelCheckpoint(best_checkpoint, monitor= 'val_acc', save_best_only= True, 
                                  mode= 'max', verbose= 1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10,mode='max')
    callbacks_list= [early_stopping,modelcheckpoint,history, csv]

    start_train =time.perf_counter() #训练开始时间
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

    save_data(predictions,taskSavePath+"predictions.pickle") #保存模型预测结果
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=classNames, digits= 5))
    
    # 打印输出模型运行的时间
    print("时间打印\n")
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
    #加载训练数据和测试数据
    basePath="/gpfs/share/home/2101211725/HFIR/DataSets/Our/"
    taskSavePath="Our_SP1/" #taskSavePath为模型训练过程中输出文件的保存位置
    trainX=load_img(basePath+"img/"+'all_img_arr_train.pickle')
    trainY = load_img(basePath+"label/"+'all_img_label_train.pickle')
    testX=load_img(basePath+"img/"+'all_img_arr_test.pickle')
    testY = load_img(basePath+"label/"+'all_img_label_test.pickle')

    classNames = ['positive', 'neutral', 'negative']
    #执行VGG19微调与评估过程
    build_model(trainX,testX,trainY,testY,classNames,taskSavePath)
