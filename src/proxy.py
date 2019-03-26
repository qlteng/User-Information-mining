# coding: utf-8

import logging
import multiprocessing
import DataConf
import DataPreprocess
import ModelConf
import ModelBuilder
import os
from utils.parse import config_parse

LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
TerminalType = ["Logger+Wifi for Android;1.0", "NexusS", "Nexus"]
TerminalPosition = ['arm','bag','waist','chest','']
Activity = ['jog','skip','stay','stDown','stUp','walk']
Person = ["person06010","person06011","person06012","person06013","person06014",\
          "person06015","person06016","person06017","person06018","person06019",\
          "person06020","person06021","person06022","person06023"]


def run(path):

    datasource, types, n_steps, n_channel, n_class, overlap, target, process_num, filter = config_parse(path)
    modelname_prefix = '_'.join([datasource, n_steps, n_channel, n_class, overlap, target, filter['phonetype'], filter['phoneposition'], filter['activity']])
    n_steps, n_channel, n_class, process_num = map(lambda x : int(x), [n_steps, n_channel, n_class, process_num])
    dataconf = DataConf.DataConf(datasource, types, n_steps, n_channel, n_class, float(overlap))
    process = DataPreprocess.DataPreprocess(dataconf, process_num, target, phonetype = filter['phonetype'], phoneposition = filter['phoneposition'], activity = filter['activity'])

    x_train, y_train, x_valid, y_valid, x_test, y_test = process.load_data(standard=False,issample=False)
    # print y_train
    if len(y_train) == 0:
        return
    '''
    #for model in ['cnn', 'vgglstm', 'lstm', 'bilstm','vgg']:
    #for model in ['cnn', 'vgglstm', 'vgg']:
    model = 'cnn'
    modelname = "%s#%s" % (model, modelname_prefix)
    modelconf = ModelConf.ModelConf(dataconf=dataconf, batch_size=400, learning_rate=0.0001, epochs=50)
    modelbuild = ModelBuilder.ModelBuilder(modelconf, modelname, allconfig['target'])
    modelbuild.train_lstm(x_train, y_train, x_valid, y_valid, figplot=True)
    modelbuild.test(x_test, y_test, ROC=False)
    '''
    record = []
    for model in ['vgglstm']:
        p = multiprocessing.Process(target=para_train, args=(model, modelname_prefix, dataconf, target, x_train, y_train, x_valid, y_valid, x_test, y_test))
        p.start()
        record.append(p)
    for process in record:
        process.join()

def para_train(model, modelname_prefix, dataconf, target, x_train, y_train, x_valid, y_valid, x_test, y_test):

    modelname = "%s#%s"% (model,modelname_prefix)
    modelconf = ModelConf.ModelConf(dataconf=dataconf, batch_size=600, learning_rate=0.0001, epochs=50)
    modelbuild = ModelBuilder.ModelBuilder(modelconf, modelname, target)
    if model == 'cnn':

        modelbuild.train_cnn(x_train, y_train, x_valid, y_valid, figplot=True)
        modelbuild.test(x_test, y_test, ROC=False)
    elif model == 'vgglstm':

        modelbuild.train_vgg_lstm(x_train, y_train, x_valid, y_valid, figplot=False)
        modelbuild.test(x_test, y_test, ROC=True)
    elif model == 'vgg':

        modelbuild.train_vgg(x_train, y_train, x_valid, y_valid, figplot=True)
        modelbuild.test(x_test, y_test, ROC=False)
    elif model == 'lstm':
        modelbuild.train_lstm(x_train, y_train, x_valid, y_valid, figplot=True)
        modelbuild.test(x_test, y_test, ROC=False)

def main():
    path = './model.conf'
    run(path)

if __name__ == '__main__':

    main()

    # for pos in TerminalPosition:
    #     allconfig['condition']['phoneposition'] = pos
    #     run(allconfig)



