# coding: utf-8

import numpy as np
from sklearn.model_selection import train_test_split
import DataConf
import DataPreprocess
import ModelConf
import ModelBuilder


if __name__ == '__main__':

    # dataconf = DataConf.DataConf('har', 'recog', 128)  # datasource,type,nsteps
    # dataconf = DataConf.DataConf('wisdm', 'recog', 128)
    dataconf = DataConf.DataConf('hasc', 'recog', 128)
    process = DataPreprocess.DataPreprocess()

    X_train, labels_train = process.read_har(data_path=dataconf.path, split="train")
    X_test, labels_test = process.read_har(data_path=dataconf.path, split="test")


    temp_data = np.concatenate((X_train, X_test), axis=0)
    temp_label = np.concatenate((labels_train, labels_test), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(temp_data, temp_label, test_size=0.25, random_state=123)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.33, random_state=456)
    print "Read HAR Done!"
    x_train, x_test, x_valid = process.standardize(x_train, x_test, x_valid)

    y_train = process.one_hot(y_train, dataconf.n_class)
    y_valid = process.one_hot(y_valid, dataconf.n_class)
    y_test  = process.one_hot(y_test, dataconf.n_class)

    cnnconf = ModelConf.ModelConf(dataconf = dataconf, batch_size = 600, learning_rate = 0.0001, epochs = 20)
    cnnmodel = ModelBuilder.ModelBuilder(cnnconf, "cnn4_filter2s1_pool2s2")
    cnnmodel.train_cnn(x_train, y_train, x_valid, y_valid, figplot = False)
    cnnmodel.test(x_test, y_test, ROC = False)

    # lstmconf = ModelConf.ModelConf(dataconf = dataconf, batch_size= 600, learning_rate= 0.0001, epochs = 40, lstm_size = 27, lstm_layer = 2)
    # lstmmodel = ModelBuilder.ModelBuilder(lstmconf, "lstm")
    # lstmmodel.train_lstm(x_train, y_train, x_valid, y_valid, figplot = True)
    # lstmmodel.test(x_test, y_test， ROC = False)


