# -*- coding: utf-8 -*-

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 100
display_step = 1
train_size = 800
step_size = 1000

# Network Parameters
n_input = 8
n_hidden_1 = 64
n_hidden_2 = 64
n_classes = 2

# convert label to numeric value
df = pd.read_csv('../input/train.csv')
le = preprocessing.LabelEncoder()
df.Sex = le.fit_transform(df.Sex)
df.Cabin = le.fit_transform(df.Cabin.astype(str))
df.Embarked = le.fit_transform(df.Embarked.astype(str))

ohe = preprocessing.OneHotEncoder(categories='auto', sparse=False)
x_np = np.array(df[['Age','Pclass','Sex','SibSp','Parch','Fare','Cabin','Embarked']].fillna(0))
y_np = ohe.fit_transform(np.array(df.Survived).reshape(-1, 1))

# divide input data to train and test
x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.2)

# tf graph input
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

# create model
def multilayer_perceptron(x, weights, biases):
    # hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# construct model
pred = multilayer_perceptron(x, weights, biases)

# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initializing the variables
init = tf.global_variables_initializer()

# loanch the graph
with tf.Session() as sess:
    sess.run(init)

    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.

        for i in range(step_size):
            # get a number of random value specified in batch_size from training data
            ind = np.random.choice(batch_size, batch_size)
            x_train_batch = x_train[ind]
            y_train_batch = y_train[ind]
            # run optimization op (backpropagation) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: x_train_batch,
                                                          y: y_train_batch})
            # compute average loss
            avg_cost += c / step_size

        # display logs per epoch step
        if epoch % display_step == 0:
            print('Epoch: %04d' % (epoch+1), 'cost=%.9f' % avg_cost)

    print('Optimization finished!')
    
    # test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('Accuracy:', accuracy.eval({x: x_test, y: y_test}))
