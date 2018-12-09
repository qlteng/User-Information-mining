# coding: utf-8

import os
import numpy as np
import logging
import multiprocessing
import tensorflow as tf
import DataConf
import DataPreprocess
import ModelConf
import ModelBuilder
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

# Person = ["person06010","person06011","person06012","person06013","person06014",
#               "person06015","person06016","person06017","person06018","person06019",
#               "person06020","person06021","person06022","person06023"]

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
    # model = 'vgglstm'
    # modelname = "%s#%s" % (model, modelname_prefix)
    # # modelname = "cnn#%s" % (modelname_prefix)
    # modelconf = ModelConf.ModelConf(dataconf=dataconf, batch_size=400, learning_rate=0.0001, epochs=50)
    # modelbuild = ModelBuilder.ModelBuilder(modelconf, modelname, allconfig['target'])
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    # modelbuild.train_vgg_lstm(x_train, y_train, x_valid, y_valid, figplot=True)
    # modelbuild.test(x_test, y_test, ROC=False)

    record = []
    for model in ['cnn','vgg','vgglstm','bilstm']:
        p = multiprocessing.Process(target=para_train, args=(model, modelname_prefix, dataconf, x_train, y_train, x_valid, y_valid, x_test, y_test))
        p.start()
        record.append(p)
    for process in record:
        process.join()

def para_train(model, modelname_prefix, dataconf,x_train, y_train, x_valid, y_valid, x_test, y_test):

    modelname = "%s#%s"% (model,modelname_prefix)
    modelconf = ModelConf.ModelConf(dataconf=dataconf, batch_size=600, learning_rate=0.0001, epochs=50)
    modelbuild = ModelBuilder.ModelBuilder(modelconf, modelname, allconfig['target'])
    if model == 'cnn':
        os.environ['CUDA_VISIBLE_DEVICES'] = '8'
        modelbuild.train_cnn(x_train, y_train, x_valid, y_valid, figplot=True)
        modelbuild.test(x_test, y_test, ROC=False)

    elif model == 'vgg':
        os.environ['CUDA_VISIBLE_DEVICES'] = '7'
        modelbuild.train_vgg(x_train, y_train, x_valid, y_valid, figplot=True)
        modelbuild.test(x_test, y_test, ROC=False)

    elif model == 'vgglstm':
        os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        modelbuild.train_vgg_lstm(x_train, y_train, x_valid, y_valid, figplot=True)
        modelbuild.test(x_test, y_test, ROC=False)
    # elif model == 'lstm':
    #
    #     modelbuild.train_lstm(x_train, y_train, x_valid, y_valid, figplot=True)
    #     modelbuild.test(x_test, y_test, ROC=False)
    elif model == 'bilstm':
        os.environ['CUDA_VISIBLE_DEVICES'] = '5'
        modelbuild.train_bilstm(x_train, y_train, x_valid, y_valid, figplot=True)
        modelbuild.test(x_test, y_test, ROC=False)

if __name__ == '__main__':

    # gender : 2; generation : 10; height : 4; weight : 4; activity : 6
    allconfig = {'datasource': 'hasc', 'types': 'recog', 'n_steps': 256, 'n_channel':6, \
              'n_class': 10, 'overlap': 0.5, 'target': 'generation', 'process_num' : 40, \
              'condition' : {'phonetype' : "", 'phoneposition' : '', 'activity' : ''}}
    # run(allconfig)
    for position in TerminalPosition:
        allconfig['condition']['phoneposition'] = position
        try:
            run(allconfig)
        except:
            logging.warning("Target %s don't have %d class on the condition of %s" % (allconfig['target'], allconfig['n_class'], allconfig['condition']['phoneposition']))



