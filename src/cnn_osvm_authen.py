# coding: utf-8

import datetime
import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
import keras
from keras.models import Model,Sequential
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape, add,Flatten
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, MaxPooling1D
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import DataConf
import DataPreprocess
import ModelConf
from utils.parse import config_parse
from utils.layer_utils import AttentionLSTM
from keras.utils.vis_utils import plot_model

LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

class Keras_ModelBuilder:

    def __init__(self, modelconf, modelname, target):

        self.conf = modelconf
        self.types = modelconf.types
        self.target = target
        self.modelname = modelname
        self.modelpath = "../model/%s/%s/%s" % (self.types,target, modelname)
        if not os.path.exists(self.modelpath):
            os.makedirs(self.modelpath)
        self.output = "../output/%s/%s/%s" % (self.types, self.target, modelname)
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        self.train_time = 0
        self.test_time = 0
        self.train_size = 10
        self.test_size = 10
        self.saver = None

    def train_cnn(self, x_train, y_train, x_valid, y_valid, figplot = False, trainable = True):
        n_channel = self.conf.n_channels
        ip = Input(shape=(n_channel, self.conf.n_steps))
        y = Permute((2, 1))(ip)
        y = Conv1D(2 * n_channel, 5, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
        y = Conv1D(2 * n_channel, 5, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
        y = MaxPooling1D(pool_size=2, strides=2, padding='same')(y)
        # y = BatchNormalization()(y)
        # y = Activation('relu')(y)
        y = Conv1D(4 * n_channel, 5, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
        y = Conv1D(4 * n_channel, 5, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
        y = MaxPooling1D(pool_size=2, strides=2, padding='same')(y)
        # y = Flatten()(y)
        y = LSTM(32, return_sequences=False, name='lstm')(y)
        y = Dense(32, activation='sigmoid')(y)
        y = Dropout(0.5)(y)
        out = None
        if trainable:
            out = Dense(self.conf.n_class, activation='softmax')(y)
        else:
            out = Dense(14, activation='softmax')(y)
        model = Model(ip, out)
        if trainable:
            plot_model(model, to_file='model1.png', show_shapes=True)
            self.run(model, x_train, y_train, x_valid, y_valid, None, "keras_cnn_lstm", figplot)
        else:
            # model.summary()
            model.load_weights("%s/model_weight.hdf5" % self.modelpath)
            model.trainable = False
            prefix_cnn = Model(inputs=model.input, outputs=model.layers[-3].output)
            self.train_svm(prefix_cnn,x_train,y_train,x_valid,y_valid)

    def train_svm(self,model,x_train,y_train,x_valid,y_valid, ROC = True):

        x_train = np.array([d.T for d in x_train])
        x_valid = np.array([d.T for d in x_valid])
        x_train = model.predict(x_train)

        x_valid = model.predict(x_valid)
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(x_train)
        y_pred_test = clf.predict(x_valid)
        y_true = []
        y_valid = np.argmax(y_valid,axis=1)
        for x in y_valid:
            if x ==1:
                y_true.append(-1)
            else:
                y_true.append(1)
        print len(y_true)
        print len(y_valid)
        print"Test accuracy: {:.6f}".format(accuracy_score(y_true, y_pred_test))
        print "confusion_matrix"
        cf_matrix = confusion_matrix(y_true, y_pred_test)
        print cf_matrix

        y_problist = clf.decision_function(x_valid)
        if ROC == True:

            y_prob = np.array(y_problist)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            FAR = 1 - tpr
            FRR = fpr
            res = abs(FAR - FRR)
            EER_index = np.argwhere(res == min(res))
            EER = (FRR[EER_index][0][0] + FAR[EER_index][0][0]) / 2
            roc_auc = auc(fpr,tpr)
            print "AUC",roc_auc
            print "FRR", cf_matrix[1][0] / float(sum(cf_matrix[1]))
            print "FAR", cf_matrix[0][1] / float(sum(cf_matrix[0]))
            print "EER", EER

            fpr = np.array(fpr)
            tpr = np.array(tpr)
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0 ,0)

            plt.plot([0, 1], [0, 1], '--', color = (0.6, 0.6, 0.6))
            plt.plot(fpr, tpr, 'b-')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positiove Rate")
            plt.title("ROC")
            plt.legend(loc = "lower right")
            plt.show()


    def run(self, model, x_train, y_train, x_valid, y_valid, cell, type, figplot = False):

        x_train = np.array([d.T for d in x_train])
        x_valid = np.array([d.T for d in x_valid])

        optm = Adam(lr=1e-3)
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
        history = LossHistory()
        start_time = datetime.datetime.now()
        model.fit(x_train, y_train,
                  batch_size=self.conf.batch_size,
                  epochs=self.conf.epochs,
                  verbose=0,
                  validation_data=(x_valid, y_valid),
                  callbacks=[history])
        self.train_time = datetime.datetime.now() - start_time
        self.train_size = len(x_train)
        weight_path = "%s/model_weight.hdf5" % self.modelpath
        if not os.path.isfile(weight_path):
            model.save_weights(weight_path)
        model.save("%s/model.h5" % self.modelpath)
        model.save_weights("%s/model_weight.hdf5"% self.modelpath)

        model.save("%s/model.h5" % self.modelpath)
        if type == 'lstm_attention':
            self.saver = model
        history.loss_plot('epoch', self.output)

    def squeeze_excite_block(self, input):
        filters = input._keras_shape[-1]  # channel_axis = -1 for TF

        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 4, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([input, se])
        return se

    def test(self, x_test, y_test, type, ROC = False):

        x_test = np.array([d.T for d in x_test])
        model = None
        if type == 'lstm_attention':
            model = self.saver
        else:
            model = load_model("%s/model.h5" % self.modelpath)
        start_time = datetime.datetime.now()
        pred = model.predict(x_test)
        self.test_time = datetime.datetime.now() - start_time
        self.test_size = len(x_test)
        pred = [x.argmax() for x in pred]
        true = [x.argmax() for x in y_test]

        print "train time", self.train_time.total_seconds()
        print "test time", self.test_time.total_seconds()
        print "train size", self.train_size
        print "test size", self.test_size

        print"Test accuracy: {:.6f}".format(accuracy_score(true, pred))
        print "Precision", precision_score(true, pred)
        print "Recall", recall_score(true, pred)
        print "f1_score", f1_score(true, pred)
        print "confusion_matrix"
        cf_matrix = confusion_matrix(true, pred)

        cf_matrix_path = "%s/cf.txt" % self.output
        cf_matrix = np.array(cf_matrix)
        np.savetxt(cf_matrix_path, cf_matrix, fmt="%d")

        res = [self.modelname, self.train_time, self.test_time, self.train_size, self.test_size, accuracy_score(true, pred), \
               precision_score(true, pred, average='weighted'),
               recall_score(true, pred, average='weighted'),
               f1_score(true, pred, average='weighted')]
        header = ["modelname", "train time", "test time", "train size", "test size", "Test accuracy", "Precision",
                  "Recall", "f1_score"]
        res_csv = "../output/%s/%s/res.csv" % (self.types, self.target)
        res = pd.DataFrame([res])
        if not os.path.exists(res_csv):
            res.to_csv(res_csv, header=header, index=False, mode="a+")
        else:
            res.to_csv(res_csv, header=False, index=False, mode="a+")

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type, save_path):
        iters = range(len(self.losses[loss_type]))

        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)

        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig("%s/acc-loss.jpg" % save_path)
        plt.show()

if __name__ == '__main__':


    model = "authen-cnn-svm"
    path = "./model.conf"
    datasource, types, n_steps, n_channel, n_class, overlap, target, process_num, filter = config_parse(path)
    modelname_prefix = '_'.join(
        [datasource, n_steps, n_channel, n_class, overlap, target, filter['phonetype'], filter['phoneposition'],
         filter['activity']])
    n_steps, n_channel, n_class, process_num = map(lambda x: int(x), [n_steps, n_channel, n_class, process_num])

    multi_dataconf = DataConf.DataConf(datasource, types, n_steps, n_channel, n_class, float(overlap))
    multi_process = DataPreprocess.DataPreprocess(multi_dataconf, process_num, target, phonetype=filter['phonetype'],
                                            phoneposition=filter['phoneposition'], activity=filter['activity'])
    x_train1, y_train1, x_valid1, y_valid1, x_test1, y_test1 = multi_process.load_data(standard=False)

    multi_modelname = "%s#%s" % (model, modelname_prefix)
    multi_modelconf = ModelConf.ModelConf(dataconf=multi_dataconf, batch_size=500, learning_rate=0.0001, epochs=300)

    multi_modelbuild = Keras_ModelBuilder(multi_modelconf, multi_modelname, target)
    multi_modelbuild.train_cnn(x_train1, y_train1, x_valid1, y_valid1,figplot=True,trainable=True)
    multi_modelbuild.test(x_test1, y_test1, 'vgg_lstm', ROC=False)

    datasource, types, n_steps, n_channel, n_class, overlap, target, process_num, filter = config_parse(path)
    target = 'binary'
    n_class = '2'
    n_steps, n_channel, n_class, process_num = map(lambda x: int(x), [n_steps, n_channel, n_class, process_num])
    binary_dataconf = DataConf.DataConf(datasource, types, n_steps, n_channel, n_class, float(overlap))
    binary_process = DataPreprocess.DataPreprocess(binary_dataconf, process_num, target, phonetype=filter['phonetype'],
                                            phoneposition=filter['phoneposition'], activity=filter['activity'])

    x_train2, y_train2, x_valid2, y_valid2, x_test2, y_test2 = binary_process.load_data(standard=False)

    datasource, types, n_steps, n_channel, n_class, overlap, target, process_num, filter = config_parse(path)
    target = 'one-class'
    n_class = '1'
    modelname_prefix = '_'.join(
        [datasource, n_steps, n_channel, n_class, overlap, target, filter['phonetype'], filter['phoneposition'],
         filter['activity']])
    n_steps, n_channel, n_class, process_num = map(lambda x: int(x), [n_steps, n_channel, n_class, process_num])
    one_dataconf = DataConf.DataConf(datasource, types, n_steps, n_channel, n_class, float(overlap))
    one_process = DataPreprocess.DataPreprocess(one_dataconf, process_num, target, phonetype=filter['phonetype'],
                                                  phoneposition=filter['phoneposition'], activity=filter['activity'])
    x_train3, y_train3, x_valid3, y_valid3, x_test3, y_test3 = one_process.load_data(standard=False)
    one_modelname = "%s#%s" % (model, modelname_prefix)
    one_modelconf = ModelConf.ModelConf(dataconf=one_dataconf, batch_size=500, learning_rate=0.0001, epochs=1000)

    one_modelbuild = Keras_ModelBuilder(one_modelconf, one_modelname, target)

    one_modelbuild.modelpath = multi_modelbuild.modelpath
    one_modelbuild.train_cnn(x_train3, y_train3, x_test2, y_test2, figplot=False, trainable=False)


