import numpy as np
import random
import logging

LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

def sample(X,Y):

    indexs = np.argwhere(Y == '0').T[0]
    neg_sample_index = np.argwhere(Y == '1').T[0]
    neg_num = len(neg_sample_index)
    pos_num = len(indexs)
    radio = float(pos_num) / float(neg_num)
    logging.info("The radio of positive and negtive is %0.1f" % radio)
    neg_sample_index = [x for x in neg_sample_index if random.random()<radio]
    indexs = np.append(indexs,neg_sample_index)
    datax= []
    datay = []
    for index in indexs:
        datax.append(X[index])
        datay.append(Y[index])
    datax = np.array(datax)
    datay = np.array(datay)
    RanSelf = np.random.permutation(datax.shape[0])
    X_ran = datax[RanSelf]
    Y_ran = datay[RanSelf]
    return X_ran,Y_ran







