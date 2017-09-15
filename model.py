import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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
                                            ichannel, self.num_channels])
            b = tf.get_variable('bias', [self.num_channels])
            h = tf.nn.conv2d(x, w, self.kstride, 'SAME', name='conv') + b
            return tf.nn.relu(h)

    def pool(self, x):
        return tf.max_pool(x,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool')

    def fully_connected(self, x, n_output, scope,
                        activation=tf.nn.relu):
        with tf.variable_scope(scope):
            isize = x.shape[-1].value
            w = tf.get_variable('weights', [isize, n_output])
            b = tf.get_variable('bias', [n_output])
            return activation(tf.matmul(x, w) + b)

    def inference(self):
        x = tf.placeholder(tf.float32, [None, 32, 32, 3], 'input')

        for i in range(self.num_channels):
            with tf.variable_scope('module_{}'.format(i)):
                x = self.conv2d(x, 'conv1')
                x = self.conv2d(x, 'conv2')
                x = self.pool(x)

        isize = np.prod([d.value for d in x.shape[-3:]])
        x = tf.reshape(x, [None, isize])
        for i in range(self.num_fc):
            scope = 'fully_connected_{}'.format(i)
            x = self.fully_connected(x, self.num_hidden, scope)

        scope = 'output'
        logits = self.fully_connected(x, 10, scope, lambda x: x)

        return logits

    def loss(self, logits):
        y = tf.placeholder(tf.float32, [None, 10], 'output')
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        loss = tf.losses.softmax_cross_entropy(logits, y, scope='cross_entropy')
        return loss, accuracy

    def train(self, loss):
        global_step = tf.get_variable('global_step', trainable=False)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        return opt.minize(loss, global_step)




def train(model, num_steps, batch_size):
    nn = model.inference()
    loss_nn, accuracy_nn = model.loss(nn)
    train_nn = model.train(loss_nn)

    sess = tf.Session()
    sess.run(tf.global_variable_initializer())

    for x in range(num_steps):
        xbatch, ybatch = mnist.train.next_batch(batch_size)
        feed_dict = {'input:0': xbatch, 'output:0': ybatch}
        calc = [loss_nn, accuracy_nn, train_nn]
        loss, accuracy, _ = sess.run(calc, feed_dict)
        print(accuracy)

if __name__ == '__main__':
    model = CNN(num_modules=2,
                num_fc=2,
                ksize=3,
                kstride=2,
                num_channels=64,
                num_hidden=100,
                learning_rate=0.01)
    train(model, 2000, 100)

