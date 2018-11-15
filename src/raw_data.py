import numpy as np
import pandas as pd
import os
from utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import MySQLdb
# get_ipython().magic(u'matplotlib inline')
import tensorflow as tf
from sklearn.model_selection import train_test_split

def standard(X):
    X_norm = (X - np.mean(X, axis=0)[None, :, :]) / np.std(X, axis=0)[None, :, :]
    return X_norm


def windows(data, size):
    start = 0
    while start < data.count() - size / 2:
        yield start, start + size
        start += (size / 2)


def segment_signal(data, window_size):
    segments = np.empty((0, window_size, data.shape[1]))
    #    labels = np.empty((0))
    #     print len(data['timestamp'])

    for (start, end) in windows(data['X'], window_size):
        # print count
        # count += 1
        x = data["X"][start:end]
        y = data["Y"][start:end]
        z = data["Z"][start:end]
        if (len(data['X'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            #            labels = np.append(labels, stats.mode(data["activity"][start:end])[0][0])
    return segments

def getdata(n_channels, window_size = 256):
    Part = 'localhost'
    User = 'root'
    Passwd = 'tql19940205'
    Database = 'qltenghasc'
    # Database='sensortest'

    db = MySQLdb.connect(Part, User, Passwd, Database)
    cursor = db.cursor()

    ActUserList = {}
    # Activity = ['jog']
    Activity = ['jog', 'skip', 'stay', 'stDown', 'stUp', 'walk']
    for One_type in Activity:
        ActUserList[One_type] = []
        sql = 'select Id from meta where Activity="'"%s"'" and TerminalPosition="wear;pants;waist;fit;right-front" ;' % One_type
        cursor.execute(sql)
        result = cursor.fetchall()
        for x in result:
            ActUserList[One_type].append(int(x[0]))
    SegmentAll = np.empty((0, window_size, n_channels))
    Labels = []

    print 'Activity Dict finished'

    for One_type in ActUserList.keys():
        i = 0
        for Id in ActUserList[One_type]:
            sql = 'select X,Y,Z from acc where MetaId=%d;' % Id
            cursor.execute(sql)
            i+=1
            if i%100==0:

                print "Now finished index :{} in {}".format(i,One_type)

            SubData = pd.DataFrame(list(cursor.fetchall()), columns=['X','Y','Z'])

            segments = segment_signal(SubData, window_size)

            # print 'segment finished'
            label = list([One_type] * int(segments.shape[0]))

            SegmentAll = np.vstack([SegmentAll, segments])

            Labels.extend(label)
            # print 'extend finished'
    # one-hot
    LabelAll = np.asarray(pd.get_dummies(np.array(Labels)), dtype=np.int8)


    cursor.close()
    db.close()


    return SegmentAll,LabelAll


def TVTsplit(X,Y):
    X_tr,X_test,Y_tr,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
    X_train,X_valid,Y_train,Y_valid=train_test_split(X_tr,Y_tr,test_size=0.33,random_state=0)
    X_train=standard(X_train)
    X_test=standard(X_test)
    X_valid=standard(X_valid)
    return X_train,Y_train,X_valid,Y_valid,X_test,Y_test

# X=pd.f
def matplot(X):
    means_ = np.zeros((X.shape[1],X.shape[2]))
    stds_ = np.zeros((X.shape[1],X.shape[2]))
    print means_.shape

    for ch in range(X.shape[2]):
        means_[:,ch] = np.mean(X[:,:,ch], axis=0)
        stds_[:,ch] = np.std(X[:,:,ch], axis=0)

    #
    df_mean = pd.DataFrame(data = means_)
    df_std = pd.DataFrame(data = stds_)
    #
    #
    #
    df_mean.hist()
    plt.show()
    #
    #
    # # In[6]:
    #
    df_std.hist()
    plt.show()

