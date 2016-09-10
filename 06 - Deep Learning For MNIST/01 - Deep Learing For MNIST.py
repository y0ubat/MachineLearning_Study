import tensorflow as tf
import input_data
import random
import matplotlib.pyplot as plt


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float', [None, 10])

dropout_rate = tf.placeholder("float")
learning_rate = 0.0000000000001
training_epochs = 1
batch_size = 100

path = './DATA/'
checkpoint_dir = "./cps/"
data = input_data.read_data_sets(path, one_hot=True)

W1 = tf.get_variable("W1", shape=[784, 512], initializer=xavier_init(784, 512))
W2 = tf.get_variable("W2", shape=[512, 256], initializer=xavier_init(512, 256))
W3 = tf.get_variable("W3", shape=[256, 312], initializer=xavier_init(256, 312))
W4 = tf.get_variable("W4", shape=[312, 100], initializer=xavier_init(312, 100))
W5 = tf.get_variable("W5", shape=[100, 10], initializer=xavier_init(100, 10))

B1 = tf.Variable(tf.zeros([512]))
B2 = tf.Variable(tf.zeros([256]))
B3 = tf.Variable(tf.zeros([312]))
B4 = tf.Variable(tf.zeros([100]))
B5 = tf.Variable(tf.zeros([10]))

_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4))
L4 = tf.nn.dropout(_L4, dropout_rate)

h = tf.add(tf.matmul(L4, W5), B5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.Session() as sess:
    saver = tf.train.Saver()

    if tf.gfile.Exists(checkpoint_dir + '/model.ckpt'):
        saver.restore(sess, checkpoint_dir + '/model.ckpt')
    else:
        init = tf.initialize_all_variables()
        sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(data.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = data.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7})
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, dropout_rate: 0.7}) / total_batch

        if epoch % 1 == 0:
            print "Epoch :", (epoch + 1), "cost =", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    r = random.randint(0, data.test.num_examples - 1)

    print "Label: ", sess.run(tf.argmax(data.test.labels[r:r + 1], 1))
    print "Prediction: ", sess.run(tf.argmax(h, 1), {X: data.test.images[r:r + 1], dropout_rate: 1.0})

    correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print "Accuracy:", accuracy.eval({X: data.test.images, Y: data.test.labels, dropout_rate: 1.0})

    saver.save(sess, checkpoint_dir + 'model.ckpt')

    plt.imshow(data.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
