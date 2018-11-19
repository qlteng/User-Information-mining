# coding: utf-8

import numpy as np
import logging
import DataConf
import DataPreprocess
import ModelConf
import ModelBuilder

LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

Person = ["person06010","person06011","person06012","person06013","person06014",
              "person06015","person06016","person06017","person06018","person06019",
              "person06020","person06021","person06022","person06023"]

TerminalType = ["Samsung;Galaxy Nexus;AndroidOS 4.1;","Logger+Wifi for  Android;1.0","Samsung;NexusS;AndroidOS 4.1;"]

TerminalPosition = ["wear;outer;chest;left","arm;right;hand","wear;pants;waist;fit;right-front",
                    "wear;pants;waist;fit;right-back","bag;position(fixed);shoulderbag","bag;position(fixed);handback"
                    "bag;position(fixed);messengerbag", "bag;position(fixed);backpack"]

Activity = ['jog','skip','stay','stDown','stUp','walk']

if __name__ == '__main__':

    # dataconf = DataConf.DataConf('har', 'recog', 128, 9, 6, 0.5)  # datasource,type,nsteps,n_channels
    # dataconf = DataConf.DataConf('wisdm', 'recog', 128, 3, 6, 0.5)

    allconfig = {'datasource': 'hasc', 'types': 'recog', 'n_steps': 128, 'n_channel':6, \
              'n_class': 6, 'overlap': 0.5, 'target': 'activity', 'process_num' : 40, \
              'condition' : {'phonetype' : '', 'phoneposition' : '', 'activity' : ''}}

    dataconf = DataConf.DataConf(datasource = allconfig['datasource'], types = allconfig['types'], n_steps = allconfig['n_steps'], \
                                 n_channels = allconfig['n_channel'], n_class = allconfig['n_class'], overlap = allconfig['overlap'])

    process = DataPreprocess.DataPreprocess(dataconf = dataconf, process_num = allconfig['process_num'], target = allconfig['target'])

    x_train, y_train, x_valid, y_valid, x_test, y_test = process.load_data(standard = True)

    modelname = '%s_%s_T_%s_P_%s_A_%s_%d_%d_%d_%0.1f' % (allconfig['datasource'], allconfig['target'], \
                allconfig['condition']['phonetype'], allconfig['condition']['phoneposition'], allconfig['condition']['activity'], \
                allconfig['n_class'], allconfig['n_steps'], allconfig['n_channel'], allconfig['overlap'])

    cnnconf = ModelConf.ModelConf(dataconf = dataconf, batch_size = 600, learning_rate = 0.0001, epochs = 40)
    cnnmodel = ModelBuilder.ModelBuilder(cnnconf, modelname)
    cnnmodel.train_cnn(x_train, y_train, x_valid, y_valid, figplot = False)
    cnnmodel.test(x_test, y_test, ROC = False)

    # lstmconf = ModelConf.ModelConf(dataconf = dataconf, batch_size= 600, learning_rate= 0.0001, epochs = 40, lstm_size = 27, lstm_layer = 2)
    # lstmmodel = ModelBuilder.ModelBuilder(lstmconf, "har_lstm")
    # lstmmodel.train_lstm(x_train, y_train, x_valid, y_valid, figplot = True)
    # lstmmodel.test(x_test, y_test, ROC = False)


