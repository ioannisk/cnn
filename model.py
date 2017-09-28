
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from random import shuffle
import os
import pickle

class CNN:
    def __init__(self,
                 ksize,
                 kstride,
                 num_channels,
                 num_hidden,
                 learning_rate):
        self.ksize = ksize
        self.kstride = kstride
        self.num_channels = num_channels
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'input')
        self.y = tf.placeholder(tf.float32, [None, 100], 'output')

    def conv2d(self, x, scope, num_channel):
        with tf.variable_scope(scope):
            ichannel = x.shape[-1].value
            w = tf.get_variable('weights', [self.ksize, self.ksize,
                                            ichannel, num_channel], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('bias', [num_channel])
            h = tf.nn.conv2d(x, w, self.kstride, 'SAME', name='conv') + b
            return tf.nn.relu(h)

    def pool(self, x):
        return tf.nn.max_pool(x,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool')

    def fully_connected(self, x, n_output, scope,
                        activation=tf.nn.relu):
        with tf.variable_scope(scope):
            isize = x.shape[-1].value
            w = tf.get_variable('weights', [isize, n_output],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('bias', [n_output])
            return activation(tf.matmul(x, w) + b)

    def inference(self):
        for i, num_channel in enumerate(self.num_channels):
            with tf.variable_scope('module_{}'.format(i)):
                x = self.conv2d(self.x, 'conv1', num_channel)
                x = self.conv2d(x, 'conv2', num_channel)
                x = self.pool(x)

        isize = np.prod([d.value for d in x.shape[-3:]])
        x = tf.reshape(x, [-1, isize])
        for i, num_h in enumerate(self.num_hidden):
            scope = 'fully_connected_{}'.format(i)
            x = self.fully_connected(x, num_h, scope)

        scope = 'output'
        logits = self.fully_connected(x, 100, scope, lambda x: x)

        return logits

    def loss_acc(self, logits):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        loss = tf.losses.softmax_cross_entropy(self.y, logits, scope='cross_entropy')
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        return loss, accuracy

    def train(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        return opt.minimize(loss, global_step)


def evaluate(model, batch_size=800, test_data):
    x_test, y_test = test_data

    logits = model.inference()
    loss, accuracy = model.loss_acc(logits)

    for i in range(0, len(x_test), batch_size):
        xbatch = x_test[i:i+batch_size]
        ybatch = y_test[i:i+batch_size]

        xbatch = np.reshape(xbatch, [-1, 32, 32, 3])
        feed_dict = {'input:0': xbatch, 'output:0': ybatch}

        calc = [loss, accuracy,  summary]
        b_loss, b_accuracy, _, b_summ = sess.run(calc, feed_dict)



def train(model, batch_size, train_data, test_data,  num_epochs):
    x_train, y_train = train_data


    logits = model.inference()
    loss, accuracy = model.loss_acc(logits)
    train = model.train(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('output/train', sess.graph)
    test_writer = tf.summary.FileWriter('output/test')

    for e in range(num_epochs):
        data = list(zip(x_train, y_train))
        shuffle(data)
        x_train, y_train = zip(*data)
        for i in range(0, len(x_train), batch_size):
            xbatch = x_train[i:i+batch_size]
            ybatch = y_train[i:i+batch_size]

            xbatch = np.reshape(xbatch, [-1, 32, 32, 3])
            feed_dict = {'input:0': xbatch, 'output:0': ybatch}

            calc = [loss, accuracy, train, summary]
            b_loss, b_accuracy, _, b_summ = sess.run(calc, feed_dict)
            if i % 1000 == 0:
                train_writer.add_summary(b_summ, i)
                print(b_accuracy, b_loss)


def one_hot(labels):
    y_dim = np.max(labels) + 1
    x_dim = len(labels)
    onehot = np.zeros(y_dim*x_dim).reshape(x_dim,y_dim)
    onehot[list(range(x_dim)), labels] = np.ones(x_dim)
    return onehot


def read_data(file_):
    with open(file_, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    xs = dict_[b'data']
    ys = dict_[b'fine_labels']
    ys = one_hot(ys)
    return (xs, ys)

if __name__ == '__main__':
    if not os.path.exists('output'):
        os.makedirs('output')
    train_data = read_data('cifar-100-python/train')
    test_data = read_data('cifar-100-python/test')

    batch_size = 128
    num_epochs = 200
    model = CNN(
                ksize=3,
                kstride=[1, 1, 1, 1],
                num_channels=[32,64],
                num_hidden=[600],
                learning_rate=0.001)
    train(model, batch_size, train_data, test_data, num_epochs=num_epochs)
    # trainer = Trainer(model, batch_size, train_data, test_data)
    # trainer.trainer()

