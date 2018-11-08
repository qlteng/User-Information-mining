# coding: utf-8

import os
import pandas as pd
import numpy as np
import DataConf


class DataPreprocess:

    def __init__(self):
        pass

    def read_har(self, data_path, split="train"):

        n_steps = 128
        path_ = os.path.join(data_path, split)
        path_signals = os.path.join(path_, "Inertial_Signals")
        label_path = os.path.join(path_, "y_" + split + ".txt")
        labels = pd.read_csv(label_path, header=None)

        channel_files = os.listdir(path_signals)
        channel_files.sort()
        n_channels = len(channel_files)

        X = np.zeros((len(labels), n_steps, n_channels))
        i_ch = 0
        for fil_ch in channel_files:

            dat_ = pd.read_csv(os.path.join(path_signals, fil_ch), delim_whitespace=True, header=None)
            X[:, :, i_ch] = dat_.as_matrix()
            i_ch += 1

        return X, labels[0].values

    def one_hot(self, labels, n_class):

        expansion = np.eye(n_class)
        y = expansion[:, labels - 1].T
        assert y.shape[1] == n_class, "Wrong number of labels!"

        return y

    def standardize(self, train, test, valid):

        X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
        X_vld = (valid - np.mean(valid, axis=0)[None, :, :]) / np.std(valid, axis=0)[None, :, :]
        X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]

        print "Standardize Done!"
        return X_train, X_test, X_vld


# if __name__ == '__main__':
#
#     dataconf = DataConf.DataConf('har','reg')
#     process = DataPreprocess()
#     X_train, labels_train = process.read_har(data_path=dataconf.path, split="train")  # train
#     X_test, labels_test = process.read_har(data_path=dataconf.path, split="test")  # test