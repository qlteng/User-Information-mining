# coding: utf-8

import os
import json
import multiprocessing
import logging
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.mysql_dump import dump_from_mysql
from utils.sample import sample

LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

class DataPreprocess:

    def __init__(self, dataconf, process_num, target, *params, **kwargs):

        self.target = target
        self.df = dataconf
        self.process_num = process_num

        # condition
        self.phonetype = kwargs.get('phonetype', '')
        self.phoneposition = kwargs.get('phoneposition', '')
        self.activity = kwargs.get('activity', '')

    def load_data(self, standard = True):

        global total_data
        global total_label

        if self.df.datasource == 'har':

            X_train, labels_train = self.read_har(data_path = self.df.path, split = "train")
            X_test, labels_test = self.read_har(data_path = self.df.path, split = "test")

            logging.info( "Read HAR done!")
            total_data = np.concatenate((X_train, X_test), axis = 0)
            total_label = np.concatenate((labels_train, labels_test), axis = 0)

        elif self.df.datasource == 'wisdm':
            pass

        elif self.df.datasource == 'hasc':
            raw_data_path = ""
            if self.df.types == 'recog':
                raw_data_path = "../data/hasc/sport_simple_data"
            elif self.df.types == 'authen':
                raw_data_path = "../data/hasc/human_data"

            if not os.path.exists(raw_data_path):
                logging.warning("Raw data path %s doesn't exist" % raw_data_path)
                logging.info("Begin to dump data from mysql")
                os.mkdir(raw_data_path)
                dump_from_mysql(raw_data_path)
                logging.info("Generate and save raw_data at %s!" % raw_data_path)
            else:
                logging.info("Load raw data from %s" % raw_data_path)

            condition_path = "%s/condition_data/%s#T_%s_P_%s_A_%s"%(self.df.path, self.target, self.phonetype, self.phoneposition, self.activity)

            if not os.path.exists(condition_path):
                logging.warning("Condition data path %s doesn't exist" % condition_path)
                os.makedirs(condition_path)

            segment_data_path = "%s/step%d_overlap%0.1f_class%d_channel%d" %(condition_path, self.df.n_steps, self.df.overlap, self.df.n_class, self.df.n_channels)

            if not os.path.exists(segment_data_path):

                logging.warning("Split data path %s doesn't exist" % segment_data_path)
                logging.info("Begin to window data with the same size")
                os.mkdir(segment_data_path)
                self.para_cut(raw_data_path, segment_data_path, self.df.n_steps, self.df.overlap, self.target,
                              self.df.n_channels, self.phonetype, self.phoneposition, self.activity)
                logging.info("Cut and save split_data at %s!" % segment_data_path)

            else:
                logging.info("Load split data from %s" % segment_data_path)


            total_data, total_label= self.read_hasc(segment_data_path, self.df.n_channels, self.df.n_steps, self.target, self.df.types, self.df.n_class)
            logging.info("Task type is %s and reset target is %s" % (self.df.types, self.target))

        RanSelf = np.random.permutation(total_data.shape[0])
        total_data = total_data[RanSelf]
        total_label = total_label[RanSelf]

        #  split users as 1:1:3 for train、valid、test
        x_train, x_test, y_train, y_test = train_test_split(total_data, total_label, test_size = 0.20, random_state = 123)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.25, random_state = 456)
        # x_train, x_valid, x_test, y_train, y_valid, y_test = [x in y for (x,y) in zip(lambda  t : total_label,[x_train, x_valid, x_test, y_train, y_valid, y_test])]
        logging.info("Split train、valid、test set")

        if standard == True:
            x_train, x_test, x_valid = self.standardize(x_train, x_test, x_valid)
            logging.info("Standardize Done!")

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def map2id(self, label, target, types):

        for i in range(len(label)):

            Person, Gender, Generation, Height, Weight, Position, Type, Mount, Activity = label[i].split('#')

            if types == 'recog':

                if target == 'activity':
                    label[i] = Activity
                elif target == 'gender':
                    label[i] = Gender
                elif target == 'generation':
                    label[i] = Generation
                elif target == 'height':
                    label[i] = Height
                elif target == 'weight':
                    label[i] = Weight

            elif types == 'authen':
                if target == 'multi':
                    label[i] = Person
                elif target == 'binary':
                    if Person == "person06010":
                        label[i] = int(0)
                    else:
                        label[i] = int(1)
        return label

    def segment(self, index, raw_data_path, segment_data_path, n_steps, overlap, target, n_channels, phonetype, phoneposition, activity):

        datafile = "%s/dump_data%d.json" % (raw_data_path, index)
        labelfile = "%s/dump_label%d.json" % (raw_data_path, index)

        with open(datafile, 'r') as f_data:
            raw_data = dict(json.load(f_data))
        with open(labelfile, 'r') as f_label:
            label = dict(json.load(f_label))

        split_data = np.empty((0, n_steps , n_channels))
        attr_labels = list()

        for x in raw_data:
            label[x] = label[x].encode()
            keys = ['person', 'gender', 'generation', 'height', 'weight', 'position', 'type', 'mount', 'activity']
            values = label[x].split('#')
            tempdict = dict(zip(keys,values))

            if self.df.types == 'recog':
                if tempdict[target] == None:
                    continue
            elif self.df.types == 'authen':
                if tempdict['person'] == None:
                    continue

            if target == 'weight':
                if tempdict['weight'] in [0, '0']:
                    continue
                if float(tempdict['weight']) <= 55:
                    tempdict['weight'] = 0
                elif float(tempdict['weight']) < 65:
                    tempdict['weight'] = 1
                elif float(tempdict['weight']) <= 75:
                    tempdict['weight'] = 2
                elif float(tempdict['weight']) > 75:
                    tempdict['weight'] = 3
                if tempdict['weight'] not in [0, 1, 2, 3]:
                    continue

            if target == 'height':
                if tempdict['height'] in [0, '0']:
                    continue
                if float(tempdict['height']) <= 160:
                    tempdict['height'] = 0
                elif float(tempdict['height']) <= 170:
                    tempdict['height'] = 1
                elif float(tempdict['height']) <= 180:
                    tempdict['height'] = 2
                elif float(tempdict['height']) > 180:
                    tempdict['height'] = 3
                if tempdict['height'] not in [0, 1, 2, 3]:
                    continue

            if target == 'generation':
                if tempdict['generation'] not in ['20;early','20;late','30;early','30;late','40;early','40;late',\
                                                  '50;early','50;late','60;early','60;late']:
                    continue
                if tempdict['generation'] == '20;early':
                    if random.random() > 0.5:
                        continue

            if tempdict['activity'] not in ['jog','skip','stay','stDown','stUp','walk']:
                continue
            if activity != '' and activity != tempdict['activity']:
                continue

            if tempdict['type'] in ['Samsung;NexusS;AndroidOS 4.1;','Samsung;Nexus S;Android OS 4.1.2']:
                tempdict['type'] = 'NexusS'
            elif tempdict['type'] in ['Samsung;Galaxy Nexus;AndroidOS 4.1;','Samsung;Galaxy Nexus;Android OS 4.1.2',\
                                      'SAMSUNG; Galaxy Nexus; Android 4.1.2']:
                tempdict['type'] = 'Nexus'
            if phonetype != '' and phonetype != tempdict['type']:
                continue

            if tempdict['position'] in ['arm;hand','arm;left;hand','arm;right;hand','arm;left;wrist','arm;right;wrist']:
                tempdict['position'] = 'arm'

            elif tempdict['position'] in ['bag','bag;position(fixed);backpack','bag;position(fixed);handback',\
                                          'bag;position(fixed);messengerbag','bag;position(fixed);shoulderbag']:
                tempdict['position'] = 'bag'

            elif tempdict['position'] in ['strap;waist;rear','waist','waist;left;pocket','wear;outer;waist;front', \
                                          'wear;pants;waist;fit;right-back','wear;pants;front','wear;pants;waist;fit;left-front',\
                                          'wear;pants;waist;fit;right-front','wear;pants;waist;fit;rigtht-front',\
                                          'wear;pants;waist;ruse;right-front']:
                tempdict['position'] = 'waist'

            elif tempdict['position'] in ['wear;outer;chest','wear;outer;chest;left']:
                tempdict['position'] = 'chest'
                
            if tempdict['position'] not in ['arm','bag','waist','chest']:
                continue
            if phoneposition != '' and phoneposition != tempdict['position']:
                continue

            label[x] = '#'.join([tempdict['person'], tempdict['gender'], tempdict['generation'], str(tempdict['height']), \
                                 str(tempdict['weight']), tempdict['position'], tempdict['type'], tempdict['mount'], tempdict['activity']])

            format_data = pd.DataFrame(list(raw_data[x]), columns=['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'])
            temp_data = np.empty((0, n_steps, n_channels))

            for (start, end) in self.windows(format_data['AX'], n_steps, overlap):

                ax = format_data['AX'][start : end]
                ay = format_data['AY'][start : end]
                az = format_data['AZ'][start : end]
                gx = format_data['GX'][start : end]
                gy = format_data['GY'][start : end]
                gz = format_data['GZ'][start : end]

                if (len(format_data['AX'][start:end]) == n_steps):
                    temp_data = np.vstack([temp_data, np.dstack([ax, ay, az, gx, gy, gz])])

            temp_label = [label[x]] * int(temp_data.shape[0])
            split_data = np.vstack([split_data, temp_data])
            attr_labels.extend(temp_label)

        attr_labels = np.array(attr_labels)
        store_data_file = "%s/split_data%d.npy" % (segment_data_path, index)
        store_label_file = "%s/split_label%d.npy" % (segment_data_path, index)

        np.save(store_data_file, split_data)
        np.save(store_label_file, attr_labels)


    def windows(self, data, n_steps, overlap):
        start = 0
        while start < data.count() - n_steps * overlap:
            yield start, start + n_steps
            start += int(n_steps * overlap)

    def para_cut(self, raw_data_path, segment_data_path, n_steps, overlap, target, n_channels, phonetype, phoneposition, activity):

        process_num = self.process_num
        record = []
        for index in xrange(0, process_num):

            p = multiprocessing.Process(target = self.segment, args=(index, raw_data_path, segment_data_path, n_steps, overlap, target, n_channels, phonetype, phoneposition, activity))
            p.start()
            record.append(p)

        for process in record:
            process.join()

    def read_hasc(self, data_path, n_channels, n_steps, target, types , n_class):

        X = np.empty((0, n_steps, n_channels))
        Y = list()
        for index in range(self.process_num):

            store_data_file = "%s/split_data%d.npy" % (data_path, index)
            store_label_file = "%s/split_label%d.npy" % (data_path, index)

            temp_data = np.load(store_data_file)
            X = np.vstack([X, temp_data])
            temp_label = np.load(store_label_file)
            temp_label = self.map2id(temp_label, target, types)
            Y.extend(temp_label)

        Y = np.array(Y)
        if target == 'binary':

            X,Y = sample(X,Y)
        Y = np.asarray(pd.get_dummies(np.array(Y)), dtype=np.int8)
        logging.info("Label has %d class"%Y[0].shape[0])
        assert Y[0].shape[0] == n_class

        return X, Y

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

        Y =  np.asarray(pd.get_dummies(np.array(labels[0].values)), dtype=np.int8)
        return X, Y

    def standardize(self, train, test, valid):

        X_train = (train - np.mean(train, axis=0)[None, :, :]) / np.std(train, axis=0)[None, :, :]
        X_vld = (valid - np.mean(valid, axis=0)[None, :, :]) / np.std(valid, axis=0)[None, :, :]
        X_test = (test - np.mean(test, axis=0)[None, :, :]) / np.std(test, axis=0)[None, :, :]
        return X_train, X_test, X_vld



