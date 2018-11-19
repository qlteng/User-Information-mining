# coding: utf-8


import tensorflow as tf
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import logging

LOG_FORMAT = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

class ModelBuilder:

    def __init__(self, modelconf, modelname):

        self.conf = modelconf
        self.types = modelconf.types
        self.saver = None
        self.modelname = modelname
        self.modelpath = "../model/%s/%s" % (self.types, modelname)
        self.output = "../output/%s" % modelname

        self.train_time = 0
        self.test_time = 0
        self.train_size = 10
        self.test_size = 10

        self.graph = tf.Graph()

        with self.graph.as_default():

            self.inputs_ = tf.placeholder(tf.float32, [ None, self.conf.n_steps, self.conf.n_channels ], name = 'inputs')
            self.labels_ = tf.placeholder(tf.float32, [ None, self.conf.n_class ], name = 'labels')
            self.keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
            self.learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

            self.logits = None
            self.cost = None
            self.optimizer = None
            self.accuracy = None
            self.initial_state = None
            self.final_state = None
            self.softmax = None

    def train_lstm(self, Xtrain, Ytrain, Xvalid, Yvalid, figplot = False):

        with self.graph.as_default():

            lstm_in = tf.transpose(self.inputs_, [1, 0, 2])  # reshape into (seq_len, N, channels)
            lstm_in = tf.reshape(lstm_in, [-1, self.conf.n_channels])  # Now (seq_len*N, n_channels)

            lstm_in = tf.layers.dense(lstm_in, self.conf.lstm_size, activation=None)
            lstm_in = tf.split(lstm_in, self.conf.n_steps, 0)

            cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.conf.lstm_size), output_keep_prob = self.keep_prob_) for
                 _ in range(self.conf.lstm_layer)], state_is_tuple=True)
            self.initial_state = cell.zero_state(self.conf.batch_size, tf.float32)

        with self.graph.as_default():
            outputs, self.final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype = tf.float32,
                                                             initial_state = self.initial_state)

            logits = tf.layers.dense(outputs[-1], self.conf.n_class, name = 'logits')
            self.logits = logits
            self.softmax = tf.nn.softmax(self.logits)
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = self.labels_))
            train_op = tf.train.AdamOptimizer(self.learning_rate_)

            gradients = train_op.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
            self.optimizer = train_op.apply_gradients(capped_gradients)

            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        self.run(Xtrain, Ytrain, Xvalid, Yvalid, cell, "lstm", figplot)

    def train_cnn(self, Xtrain, Ytrain, Xvalid, Yvalid, figplot = False):

        with self.graph.as_default():
            # (batch, 128, 9) --> (batch, 64, 18)
            conv1 = tf.layers.conv1d(inputs = self.inputs_, filters = 2 * self.conf.n_channels, kernel_size = 2, strides = 1,
                                     padding = 'same', activation = tf.nn.relu)
            max_pool_1 = tf.layers.max_pooling1d(inputs = conv1, pool_size = 2, strides = 2, padding = 'same')

            # (batch, 64, 18) --> (batch, 32, 36)
            conv2 = tf.layers.conv1d(inputs = max_pool_1, filters = 4 * self.conf.n_channels, kernel_size = 2, strides = 1,
                                     padding = 'same', activation = tf.nn.relu)
            max_pool_2 = tf.layers.max_pooling1d(inputs = conv2, pool_size = 2, strides = 2, padding = 'same')

            # (batch, 32, 36) --> (batch, 16, 72)
            conv3 = tf.layers.conv1d(inputs = max_pool_2, filters = 8 * self.conf.n_channels, kernel_size = 2, strides = 1,
                                     padding = 'same', activation = tf.nn.relu)
            max_pool_3 = tf.layers.max_pooling1d(inputs = conv3, pool_size = 2, strides = 2, padding = 'same')

            # (batch, 16, 72) --> (batch, 8, 144)
            conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=16 * self.conf.n_channels, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')


        with self.graph.as_default():

            flat = tf.reshape(max_pool_4, ( -1, self.conf.n_steps * self.conf.n_channels ))
            flat = tf.nn.dropout(flat, keep_prob = self.keep_prob_)
            self.logits = tf.layers.dense(flat, self.conf.n_class)
            self.softmax = tf.nn.softmax(self.logits)

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels_))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_).minimize(self.cost)
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        self.run(Xtrain, Ytrain, Xvalid, Yvalid, None, "cnn", figplot)

    def train_cnn_rnn(self, Xtrain, Ytrain, Xvalid, Yvalid, figplot = False):
        pass

    def train_bilstm(self, Xtrain, Ytrain, Xvalid, Yvalid, figplot = False):
        pass

    def run(self, Xtrain, Ytrain, Xvalid, Yvalid, cell, type, figplot = False):

        validation_acc = []
        validation_loss = []

        train_acc = []
        train_loss = []

        with self.graph.as_default():
            self.saver = tf.train.Saver()

        start_time = datetime.datetime.now()

        with tf.Session(graph = self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1

            for e in range(1, self.conf.epochs + 1):

                if type == "lstm":
                    state = sess.run(self.initial_state)
                RanSelf = np.random.permutation(Xtrain.shape[0])
                X_ran = Xtrain[RanSelf]
                Y_ran = Ytrain[RanSelf]

                for x, y in self.get_batches(X_ran, Y_ran):

                    if type == "lstm":

                        feed = {self.inputs_: x, self.labels_: y, self.keep_prob_: 0.5,
                                self.initial_state: state, self.learning_rate_: self.conf.learning_rate}

                        loss, _, state, acc = sess.run([self.cost, self.optimizer, self.final_state, self.accuracy],
                                                       feed_dict = feed)
                    elif type == "cnn":

                        feed = {self.inputs_: x, self.labels_: y, self.keep_prob_: 0.5,
                                self.learning_rate_: self.conf.learning_rate}
                        loss, _, acc = sess.run([self.cost, self.optimizer, self.accuracy], feed_dict=feed)

                    train_acc.append(acc)
                    train_loss.append(loss)

                    if (iteration % 5 == 0):
                        print("Epoch: {}/{}".format(e, self.conf.epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:.6f}".format(acc))

                    if (iteration % 10 == 0):

                        if type == "lstm":
                            val_state = sess.run(cell.zero_state(self.conf.batch_size, tf.float32))

                        val_acc_ = []
                        val_loss_ = []
                        for x_v, y_v in self.get_batches(Xvalid, Yvalid):

                            if type == "lstm":

                                feed = {self.inputs_: x_v, self.labels_: y_v, self.keep_prob_: 1.0, self.initial_state: val_state}

                                loss_v, state_v, acc_v = sess.run([self.cost, self.final_state, self.accuracy], feed_dict=feed)

                            elif type == "cnn":

                                feed = {self.inputs_: x_v, self.labels_: y_v, self.keep_prob_: 1.0}
                                loss_v, acc_v = sess.run([self.cost, self.accuracy], feed_dict=feed)

                            val_acc_.append(acc_v)
                            val_loss_.append(loss_v)

                        print("Epoch: {}/{}".format(e, self.conf.epochs),
                              "Iteration: {:d}".format(iteration),
                              "Validation loss: {:6f}".format(np.mean(val_loss_)),
                              "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                        maxvalideacc = 0
                        validation_acc.append(np.mean(val_acc_))
                        validation_loss.append(np.mean(val_loss_))
                        if e > 0.8 * self.conf.epochs:
                            if maxvalideacc < np.mean(val_acc_):
                                maxvalideacc = np.mean(val_acc_)
                                self.saver.save(sess, '%s/model.ckpt' % self.modelpath)
                    iteration += 1

        self.train_time = datetime.datetime.now() - start_time
        self.train_size = len(Xtrain)

        if figplot == True:
            self.plot(iteration, train_loss, train_acc, validation_loss, validation_acc)

    def plot(self, iter, train_loss, train_acc, valid_loss, valid_acc):

        t = np.arange(iter - 1)

        plt.figure(figsize = (6,6))
        plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(valid_loss), 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Loss")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

        plt.figure(figsize = (6,6))
        plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], np.array(valid_acc), 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Accuray")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

    def get_batches(self, X, Y):

        batch_size = self.conf.batch_size
        n_batches = len(X) // batch_size
        X, y = X[:n_batches * batch_size], Y[:n_batches * batch_size]

        for b in range(0, len(X), batch_size):
            yield X[b:b + batch_size], y[b:b + batch_size]

    def test(self, X_test, y_test, ROC = False):

        test_acc = []
        y_plist = []
        y_truelist = []
        y_problist = []

        start_time = datetime.datetime.now()

        with tf.Session(graph = self.graph) as sess:

            self.saver.restore(sess, tf.train.latest_checkpoint(self.modelpath))
            for x_t, y_t in self.get_batches(X_test, y_test):

                feed = { self.inputs_: x_t, self.labels_: y_t, self.keep_prob_: 1 }
                y_p = tf.argmax(self.logits, 1)
                y_true = np.argmax(y_t, 1)
                batch_acc, y_pred, y_pred_prob = sess.run([self.accuracy, y_p, self.softmax], feed_dict=feed)
                y_plist.extend(y_pred)
                y_truelist.extend(y_true)
                y_problist.extend(y_pred_prob)
                test_acc.append(batch_acc)

            self.test_time = datetime.datetime.now() - start_time
            self.test_size = len(X_test)

            print "train time", self.train_time
            print "test time", self.test_time
            print "train size", self.train_size
            print "test size", self.test_size

            print"Test accuracy: {:.6f}".format(np.mean(test_acc))
            print "Precision", precision_score(y_truelist, y_plist, average='weighted')
            print "Recall", recall_score(y_truelist, y_plist, average='weighted')
            print "f1_score", f1_score(y_truelist, y_plist, average='weighted')
            print "confusion_matrix"
            cf_matrix = confusion_matrix(y_truelist, y_plist)
            print cf_matrix

            if ROC == True:

                y_prob = np.array(y_problist)
                fpr, tpr, thresholds = roc_curve(y_truelist, y_prob[:, 1])
                FAR = 1 - tpr
                FRR = fpr
                res = abs(FAR - FRR)
                EER_index = np.argwhere(res == min(res))
                EER = (FRR[EER_index][0][0] + FAR[EER_index][0][0]) / 2
                roc_auc = auc(fpr,tpr)
                print roc_auc
                print "FRR", cf_matrix[1][0] / float(sum(cf_matrix[1]))
                print "FAR", cf_matrix[0][1] / float(sum(cf_matrix[0]))
                print "EER", EER

                fpr = np.array(fpr)
                tpr = np.array(tpr)
                fpr = np.insert(fpr, 0, 0)
                tpr = np.insert(tpr, 0 ,0)

                plt.plot([0, 1], [0, 1], '--', color = (0.6, 0.6, 0.6))
                plt.plot(fpr, tpr, 'b-')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positiove Rate")
                plt.title("ROC")
                plt.legend(loc = "lower right")
                plt.show()


