import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


class CNN:
    def __init__(self,
                 num_modules,
                 num_fc,
                 ksize,
                 kstride,
                 num_channels,
                 num_hidden,
                 learning_rate):
        self.num_modules = num_modules
        self.ksize = ksize
        self.kstride = kstride
        self.num_channels = num_channels
        self.num_fc = num_fc
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate

    def conv2d(self, x, scope):
        with tf.variable_scope(scope):
            ichannel = x.shape[-1].value
            w = tf.get_variable('weights', [self.ksize, self.ksize,
                                            ichannel, self.num_channels], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('bias', [self.num_channels])
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
            w = tf.get_variable('weights', [isize, n_output], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('bias', [n_output])
            return activation(tf.matmul(x, w) + b)

    def inference(self):
        g = tf.get_default_graph()
        x = g.get_tensor_by_name('input:0')

        for i in range(self.num_channels):
            with tf.variable_scope('module_{}'.format(i)):
                x = self.conv2d(x, 'conv1')
                x = self.conv2d(x, 'conv2')
                # x = self.pool(x)

        isize = np.prod([d.value for d in x.shape[-3:]])
        x = tf.reshape(x, [-1, isize])
        for i in range(self.num_fc):
            scope = 'fully_connected_{}'.format(i)
            x = self.fully_connected(x, self.num_hidden, scope)

        scope = 'output'
        logits = self.fully_connected(x, 10, scope, lambda x: x)

        return logits

    def loss(self, logits):
        g = tf.get_default_graph()
        y = g.get_tensor_by_name('output:0')
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        loss = tf.losses.softmax_cross_entropy(y, logits, scope='cross_entropy')
        return loss, accuracy

    def train(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        return opt.minimize(loss, global_step)


def train(model, num_steps, batch_size, mnist):
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], 'input')
    y = tf.placeholder(tf.float32, [None, 10], 'output')
    nn = model.inference()
    loss_nn, accuracy_nn = model.loss(nn)
    train_nn = model.train(loss_nn)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(num_steps):
        xbatch, ybatch = mnist.train.next_batch(batch_size)
        xbatch = np.reshape(xbatch, [-1, 28,28,1])
        feed_dict = {'input:0': xbatch, 'output:0': ybatch}
        calc = [loss_nn, accuracy_nn, train_nn]
        loss, accuracy, _ = sess.run(calc, feed_dict)
        if i % 100 == 0:
            print(accuracy, loss)


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    model = CNN(num_modules=2,
                num_fc=2,
                ksize=3,
                kstride=[1, 2, 2, 1],
                num_channels=20,
                num_hidden=300,
                learning_rate=0.05)
    train(model, 20000, 100, mnist)

