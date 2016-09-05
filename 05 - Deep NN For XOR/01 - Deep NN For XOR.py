import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True)

x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32,name="X")
Y = tf.placeholder(tf.float32,name="Y")

W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0),name='W1')
W2 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0),name='W2')
W3 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0),name='W3')
W4 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0),name='W4')
W5 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0),name='W5')
W6 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0),name='W6')
W7 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0),name='W7')

b1 = tf.Variable(tf.zeros([5]), name='B1')
b2 = tf.Variable(tf.zeros([5]), name='B2')
b3 = tf.Variable(tf.zeros([5]), name='B3')
b4 = tf.Variable(tf.zeros([5]), name='B4')
b5 = tf.Variable(tf.zeros([5]), name='B5')
b6 = tf.Variable(tf.zeros([1]), name='B6')
b7 = tf.Variable(tf.zeros([1]), name='B7')

with tf.name_scope('layer1') as scope:
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
with tf.name_scope('layer2') as scope:
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
with tf.name_scope('layer3') as scope:
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
with tf.name_scope('layer4') as scope:
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
with tf.name_scope('layer5') as scope:
    L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
with tf.name_scope('layer6') as scope:
    L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
with tf.name_scope('last') as scope:
    h = tf.sigmoid(tf.matmul(L6, W7) + b7)

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h))
    cost_summ = tf.scalar_summary("cost", cost)

with tf.name_scope('train') as scope:
    rate = tf.Variable(0.005)
    optimizer = tf.train.GradientDescentOptimizer(rate)
    train = optimizer.minimize(cost)


w1_hist = tf.histogram_summary("W1", W1)
w2_hist = tf.histogram_summary("W2", W2)
w3_hist = tf.histogram_summary("W3", W3)
w4_hist = tf.histogram_summary("W4", W4)
w5_hist = tf.histogram_summary("W5", W5)
w6_hist = tf.histogram_summary("W6", W6)
w7_hist = tf.histogram_summary("W7", W7)

b1_hist = tf.histogram_summary("B1", b1)
b2_hist = tf.histogram_summary("B2", b2)
b3_hist = tf.histogram_summary("B3", b3)
b4_hist = tf.histogram_summary("B4", b4)
b5_hist = tf.histogram_summary("B5", b5)
b6_hist = tf.histogram_summary("B6", b6)
b7_hist = tf.histogram_summary("B7", b7)

y_hist = tf.histogram_summary("Y", Y)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph)
    merged = tf.merge_all_summaries()

    for step in range(30000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)
            print step,sess.run(cost, feed_dict={X: x_data, Y: y_data})
            sess.run([W1, W2,W3,W4,W5,W6,W7])

    correct_prediction = tf.equal(tf.floor(h+0.5), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print  sess.run([h, tf.floor(h+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})

    print 'Accuracy :', accuracy.eval({X:x_data, Y:y_data})