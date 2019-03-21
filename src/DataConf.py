# coding: utf-8

import logging
LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class DataConf:

    def __init__(self, datasource, types, n_steps, n_channels, n_class, overlap):

        self.datasource = datasource
        self.types = types
        self.n_steps = n_steps
        self.n_channels = n_channels
        self.n_class = n_class
        self.overlap = overlap
        self.path = None

        if self.datasource == "har":
            self.path = '../data/har/'
            self.n_steps = 128
            self.n_class = 6
            self.n_channels = 6

        elif self.datasource == 'wisdm':
            pass

        elif self.datasource == 'hasc':

            self.path = '../data/hasc'

        else:
            logging.warning("Datasource doesn't exist!")
            return -1

        logging.info("Data config succeed")