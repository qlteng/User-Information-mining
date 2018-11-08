
# coding: utf-8

# # HAR LSTM training 

# In[1]:

# Imports
import numpy as np
import os
from utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix



# ## Prepare data

# In[2]:

X_train, labels_train, list_ch_train = read_data(data_path="../data/", split="train") # train
X_test, labels_test, list_ch_test = read_data(data_path="../data/", split="test") # test

assert list_ch_train == list_ch_test, "Mistmatch in channels!"


all_data = np.concatenate((X_train,X_test), axis = 0)
all_label=np.concatenate((labels_train,labels_test), axis = 0)

X_tr, X_test, lab_tr, lab_test = train_test_split(all_data, all_label,test_size=0.25, random_state=123)
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_tr,lab_tr,
                                                test_size=0.33, random_state=456)
# Standardize
X_tr, X_test,X_vld = standardize(X_tr, X_test,X_vld)

# Train/Validation Split

# In[4]:


# One-hot encoding:

# In[5]:

y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(lab_test)

# ### Hyperparameters

# In[6]:

# Imports
import tensorflow as tf
import datetime

lstm_size = 27         # 3 times the amount of channels
lstm_layers = 2        # Number of layers
batch_size = 600       # Batch size
seq_len = 128          # Number of steps
learning_rate = 0.0001  # Learning rate (default is 0.001)
epochs = 15

# Fixed
n_classes = 6
n_channels = 9


# ### Construct the graph
# Placeholders

# In[7]:

graph = tf.Graph()
start_time = datetime.datetime.now()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')


# Construct inputs to LSTM

# In[8]:

with graph.as_default():
    # Construct the LSTM inputs and LSTM cells
    lstm_in = tf.transpose(inputs_, [1,0,2]) # reshape into (seq_len, N, channels)
    lstm_in = tf.reshape(lstm_in, [-1, n_channels]) # Now (seq_len*N, n_channels)
    
    # To cells
    lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None) # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?
    # print "++++"
    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, seq_len, 0)
    
    # Add LSTM layers
    cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(lstm_size), output_keep_prob=keep_prob_) for _ in
         range(2)], state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)




# Define forward pass, cost function and optimizer:

# In[9]:

with graph.as_default():
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                     initial_state = initial_state)
    
    # We only need the last output tensor to pass into a classifier
    logits = tf.layers.dense(outputs[-1], n_classes, name='logits')
    
    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    #optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping
    
    # Grad clipping
    train_op = tf.train.AdamOptimizer(learning_rate_)

    gradients = train_op.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    optimizer = train_op.apply_gradients(capped_gradients)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# ### Train the network
# In[11]:

validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    for e in range(1, epochs + 1):
        # Initialize
        state = sess.run(initial_state)
        RanSelf = np.random.permutation(X_tr.shape[0])
        X_ran = X_tr[RanSelf]
        Y_ran = y_tr[RanSelf]
        # Loop over batches
        for x, y in get_batches(X_ran, Y_ran, batch_size):

            # Feed dictionary
            feed = {inputs_: x, labels_: y, keep_prob_: 0.5,
                    initial_state: state, learning_rate_: learning_rate}

            loss, _, state, acc = sess.run([cost, optimizer, final_state, accuracy],
                                           feed_dict=feed)

            train_acc.append(acc)
            train_loss.append(loss)

            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))

            # Compute validation loss at every 25 iterations
            if (iteration % 20 == 0):

                # Initiate for validation set
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))

                val_acc_ = []
                val_loss_ = []
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0, initial_state: val_state}

                    # Loss
                    loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict=feed)

                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)

                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                # Store
                maxvalideacc = 0
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
                if e > 10:

                    if maxvalideacc < np.mean(val_acc_):
                        maxvalideacc = np.mean(val_acc_)
                        name = "./checkpoints-LSTM/har.ckpt"
                        saver.save(sess, name)

            # Iterate
            iteration += 1

            # if e%2==0 & e/2>1:
    # name = "checkpoints-crnn/har.ckpt"
    # saver.save(sess, name)
train_time = (datetime.datetime.now() - start_time)
print train_time.seconds
# In[14]:

# va = np.array(validation_acc)
# vl = np.array(validation_loss)
#
# ta =np.array(train_acc)
# tl = np.array(train_loss)

# Plot training and test loss
# t = np.arange(iteration)+1
# np.save('va.npy',va)
# np.save('vl.npy',vl)
# np.save('ta.npy',ta)
# np.save('tl.npy',tl)
# np.save('t.npy',t)



# plt.figure(figsize = (6,6))
# plt.plot(t, np.array(train_loss), 'r-', t[t%20==0], np.array(validation_loss), 'b*')
# plt.xlabel("iteration")
# plt.ylabel("Loss")
# plt.legend(['train', 'validation'], loc='upper right')
# plt.savefig('HAR_loss_cnn_lstm')
# plt.show()
#
#
# # In[15]:
#
# # Plot Accuracies
# plt.figure(figsize = (6,6))
#
# plt.plot(t, np.array(train_acc), 'r-', t[t%20==0], validation_acc, 'b*')
# plt.xlabel("iteration")
# plt.ylabel("Accuray")
# plt.legend(['train', 'validation'], loc='upper right')
# plt.savefig('HAR_acc_cnn_lstm')
# plt.show()


# ## Evaluate on test set

# In[16]:
test_acc = []
y_plist=[]
y_truelist=[]
with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints-LSTM'))
    for x_t, y_t in get_batches(X_test, y_test, batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1}

        y_p = tf.argmax(logits, 1)
        y_true = np.argmax(y_t, 1)
        batch_acc ,y_pred= sess.run([accuracy, y_p],feed_dict=feed)
        y_plist.extend(y_pred)
        y_truelist.extend(y_true)
        test_acc.append(batch_acc)

    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))
    print "Precision", precision_score(y_truelist, y_plist, average='weighted')
    print "Recall", recall_score(y_truelist, y_plist, average='weighted')
    print "f1_score", f1_score(y_truelist, y_plist, average='weighted')
    print "confusion_matrix"
    print confusion_matrix(y_truelist, y_plist)

