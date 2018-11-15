# coding: utf-8


class DataConf:

    def __init__(self, datasource, types, n_steps):

        self.datasource = datasource
        self.types = types
        self.n_steps = n_steps

        if self.datasource == "har":
            self.path = '../data/har/'
            self.n_steps = 128
            self.n_class = 6
            self.n_channels = 9

        elif self.datasource == 'wisdm':
            pass

        elif self.datasource == 'hasc':

            pass

        else:
            print "Datasource Doesn't Exist!"
            return -1

        if self.types not in ['recog' ,'authen']:
            print "Types Doesn't Match!"
            return -1