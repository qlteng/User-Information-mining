# coding: utf-8


class ModelConf:

    def __init__(self, dataconf, batch_size, learning_rate, epochs, lstm_size = 27, lstm_layer = 2):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lstm_size = lstm_size
        self.lstm_layer = lstm_layer

        self.n_steps = dataconf.n_steps
        self.n_class = dataconf.n_class
        self.n_channels = dataconf.n_channels
        self.types = dataconf.types