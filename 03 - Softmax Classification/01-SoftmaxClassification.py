import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt',unpack=True,dtype='float32')

x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

X = tf.placeholder('float',[None,3])
Y = tf.placeholder('float',[None,4])

W = tf.Variable(tf.zeros([3,4]))
h = tf.nn.softmax(tf.matmul(X,W))

learning_rate = 0.1
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(h), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(10001):
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
        if step % 1000 == 0:
            print step, sess.run(cost,feed_dict={X:x_data,Y:y_data}), sess.run(W)

    s = sess.run(h, feed_dict={X:[[1,20,15]]})
    print "a+ :", s, sess.run(tf.arg_max(s,1))

    a = sess.run(h, feed_dict={X: [[1, 7, 7]]})
    print "a : ", a, sess.run(tf.arg_max(a, 1))

    b = sess.run(h, feed_dict={X:[[1,7,5]]})
    print "b : ", b, sess.run(tf.arg_max(b,1))

    c = sess.run(h, feed_dict={X: [[1, 1, 2]]})
    print "c : ", a, sess.run(tf.arg_max(c, 1))

    all = sess.run(h, feed_dict={X: [[1, 20, 15],[1,7,7],[1,7,5],[1,1,2]]})
    print "all :", s, sess.run(tf.arg_max(all, 1))

