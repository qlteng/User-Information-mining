# coding: utf-8

import numpy as np
import logging
import multiprocessing
import DataConf
import DataPreprocess
import ModelConf
import ModelBuilder

LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


TerminalType = ["Logger+Wifi for Android;1.0", "NexusS", "Nexus"]
TerminalPosition = ['arm','bag','waist','chest','']

# TerminalPosition = ["wear;outer;chest;left","arm;right;hand","wear;pants;waist;fit;right-front",
#                     "wear;pants;waist;fit;right-back","bag;position(fixed);shoulderbag","bag;position(fixed);handback"
#                     "bag;position(fixed);messengerbag", "bag;position(fixed);backpack"]

Activity = ['jog','skip','stay','stDown','stUp','walk']


def run(allconfig):

    dataconf = DataConf.DataConf(datasource=allconfig['datasource'], types=allconfig['types'],
                                 n_steps=allconfig['n_steps'], \
                                 n_channels=allconfig['n_channel'], n_class=allconfig['n_class'],
                                 overlap=allconfig['overlap'])

    process = DataPreprocess.DataPreprocess(dataconf=dataconf, process_num=allconfig['process_num'],
                                            target=allconfig['target'], \
                                            phonetype=allconfig['condition']['phonetype'],
                                            phoneposition=allconfig['condition']['phoneposition'])

    x_train, y_train, x_valid, y_valid, x_test, y_test = process.load_data(standard=False)
    if len(x_train) == 0:
        return

    modelname_prefix = '%s_%s_T_%s_P_%s_A_%s_%d_%d_%d_%0.1f' % (allconfig['datasource'], allconfig['target'], \
                                                        allconfig['condition']['phonetype'],
                                                         allconfig['condition']['phoneposition'],
                                                         allconfig['condition']['activity'], \
                                                         allconfig['n_class'], allconfig['n_steps'],
                                                         allconfig['n_channel'], allconfig['overlap'])

    # for model in ['cnn', 'vgglstm', 'lstm', 'bilstm','vgg']:
    # for model in ['cnn', 'vgglstm', 'vgg']:
    model = 'lstm'
    modelname = "%s#%s" % (model, modelname_prefix)
    modelconf = ModelConf.ModelConf(dataconf=dataconf, batch_size=400, learning_rate=0.0001, epochs=50)
    modelbuild = ModelBuilder.ModelBuilder(modelconf, modelname, allconfig['target'])
    modelbuild.train_lstm(x_train, y_train, x_valid, y_valid, figplot=True)
    modelbuild.test(x_test, y_test, ROC=False)

#     record = []
#     for model in ['cnn','vgglstm','vgg']:
#         p = multiprocessing.Process(target=para_train, args=(model, modelname_prefix, dataconf, x_train, y_train, x_valid, y_valid, x_test, y_test))
#         p.start()
#         record.append(p)
#     for process in record:
#         process.join()
# #
# def para_train(model, modelname_prefix, dataconf,x_train, y_train, x_valid, y_valid, x_test, y_test):
#
#     modelname = "%s#%s"% (model,modelname_prefix)
#     modelconf = ModelConf.ModelConf(dataconf=dataconf, batch_size=600, learning_rate=0.0001, epochs=100)
#     modelbuild = ModelBuilder.ModelBuilder(modelconf, modelname, allconfig['target'])
#     if model == 'cnn':
#
#         modelbuild.train_cnn(x_train, y_train, x_valid, y_valid, figplot=True)
#         modelbuild.test(x_test, y_test, ROC=False)
#     elif model == 'vgglstm':
#
#         modelbuild.train_vgg_lstm(x_train, y_train, x_valid, y_valid, figplot=True)
#         modelbuild.test(x_test, y_test, ROC=False)
#     elif model == 'vgg':
#
#         modelbuild.train_vgg(x_train, y_train, x_valid, y_valid, figplot=True)
#         modelbuild.test(x_test, y_test, ROC=False)

if __name__ == '__main__':


    allconfig = {'datasource': 'hasc', 'types': 'recog', 'n_steps': 256, 'n_channel':6, \
              'n_class': 6, 'overlap': 0.5, 'target': 'activity', 'process_num' : 40, \
              'condition' : {'phonetype' : "", 'phoneposition' : '', 'activity' : ''}}
    for position in TerminalPosition:
        allconfig['condition']['phoneposition'] = position
        run(allconfig)



