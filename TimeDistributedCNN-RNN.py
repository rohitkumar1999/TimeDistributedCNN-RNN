#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


import tensorlayer as tl


# In[ ]:


import numpy as np


# In[ ]:


n_height = 64
n_width = 64
n_steps = 50
alpha = 0.01
learning_rate = 0.001
n_classes = 26
n_epochs = 40
channels = 1
batch_size = 80


# In[ ]:


x_train = np.load('path_to_x_train.npy')


# In[ ]:


y_train = np.load('path_to_y_train.npy')


# In[ ]:


x_train.shape, y_train.shape


# In[ ]:


x_test = np.load('path_to_x_test.npy')
y_test = np.load('path_to_y_test.npy')


# In[ ]:


x_test.shape, y_test.shape


# In[ ]:


x = np.vstack((x_train, x_test))
_y = np.hstack((y_train, y_test))


# In[ ]:


x.shape, _y.shape


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:


x = np.array(x, dtype=np.float32)


# In[ ]:


x = x/255


# In[ ]:


x.shape+(1,)


# In[ ]:


x = x.reshape(x.shape+(1,))


# In[ ]:


sss = StratifiedShuffleSplit(test_size=26)
for train_index, test_index in sss.split(x, _y):
    x_train, y_train = x[train_index], _y[train_index]
    x_test, y_test = x[test_index], _y[test_index]


# In[ ]:


x_train.shape, x_test.shape


# In[ ]:


x_train.shape, y_train.shape


# In[ ]:


def next_batch(batch_size = batch_size, x_train = x_train, y_train = y_train):
    strtix = 0
    endix = batch_size
    while endix <= len(x_train):
        yield x_train[strtix:endix], y_train[strtix:endix]
        strtix = endix
        endix+=batch_size


# In[ ]:


for a,b in next_batch():
    print(np.unique(b, return_counts=True))


# In[ ]:


tf.reset_default_graph()


# In[ ]:


with tf.variable_scope('Input'):
    X = tf.placeholder(dtype=tf.float32, name='X', shape=[None,n_steps, n_height, n_width,channels])
    y = tf.placeholder(dtype=tf.int32, name='y', shape=[None])


# In[ ]:


with tf.variable_scope('CNN'):
    X_layer = tl.layers.InputLayer(X, name='X_layer')
    cnn1 = tl.layers.TimeDistributedLayer(prev_layer=X_layer, layer_class=tl.layers.Conv2dLayer, layer_args={'act':tf.nn.relu, 'shape':(7,7,1,3), 'strides':(1,2,2,1),'name':'cnn1-layer', 'padding':'SAME'})
    cnn2 = tl.layers.TimeDistributedLayer(prev_layer=cnn1, layer_class=tl.layers.Conv2dLayer, layer_args={'act':tf.nn.relu, 'shape':(5,5,3,1), 'strides':(1,3,3,1),'name':'cnn2-layer', 'padding':'VALID'})
    avg_pool1 = tl.layers.TimeDistributedLayer(prev_layer=cnn2, layer_class=tl.layers.MeanPool2d, layer_args={'filter_size':(4,4), 'strides':(1,1), 'padding':'SAME','name':'avg_pool1'})
    flatten = tl.layers.TimeDistributedLayer(prev_layer=avg_pool1, layer_class=tl.layers.FlattenLayer, layer_args={'name':'flatten1'})
    cnn_output = flatten.outputs


# In[ ]:


with tf.variable_scope('RNN', initializer=tf.contrib.layers.variance_scaling_initializer()):
    cell = tf.nn.rnn_cell.GRUCell(activation=tf.nn.elu, dtype=tf.float32,num_units=30, name='gru')
    outputs, states = tf.nn.dynamic_rnn(cell=cell, dtype= tf.float32, inputs=cnn_output)
    flatten_outputs = tf.layers.flatten(inputs=outputs,name='flatten_output')
    training = tf.placeholder_with_default(input=False, name='training', shape=())
    dropout1 = tf.layers.dropout(rate=0.3, name='dropout1', inputs=flatten_outputs, training=training)
    hidden1 = tf.layers.dense(activation = tf.nn.elu, units=100, inputs= dropout1, name='dnn-output1', kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            kernel_regularizer = tf.contrib.layers.l1_regularizer(scale=alpha))
    dropout2 = tf.layers.dropout(rate=0.2, name='dropout2', inputs = hidden1, training=training)
    stop_gradient = tf.stop_gradient(input=dropout2,name='stop_gradient')
    hidden2 = tf.layers.dense(activation = None, units = n_classes, inputs = stop_gradient, name='dnn-output3', kernel_initializer =tf.contrib.layers.variance_scaling_initializer(), 
                              kernel_regularizer= tf.contrib.layers.l1_regularizer(scale=alpha))
    prediction = tf.nn.softmax(logits = hidden2)


# In[ ]:


with tf.variable_scope('loss'):
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hidden2, labels= y)
    loss = tf.reduce_mean(xentropy) + regularization_loss
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

with tf.variable_scope('performance'):
    predictions = tf.nn.in_top_k(k=1, name='predictions', targets=y, predictions=prediction)
    accuracy = tf.reduce_mean(tf.cast(predictions, dtype=tf.float32))
    
with tf.variable_scope('training'):
    training_op = optimizer.minimize(loss)


# In[ ]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()
restore = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    restore.restore(sess,'./model/TimeDistributedCNN-RNN.ckpt')
    try:
        for epoch in range(n_epochs):
            for x_batch, y_batch in next_batch():
                sess.run(training_op, feed_dict = {X:x_batch, y:y_batch, training:True})
            acc_train = accuracy.eval(feed_dict={X:x_train, y:y_train})
            acc_test = accuracy.eval(feed_dict = {X:x_test, y:y_test})
            print('epoch:',epoch, 'Training acc:', acc_train, 'Testing_acc:', acc_test)
    finally:
        saver.save(sess, './model/updated_TimeDistributedCNN-RNN.ckpt')

