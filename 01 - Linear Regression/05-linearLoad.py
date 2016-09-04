import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt',unpack=True,dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]


print 'x', x_data
print 'y', y_data
print len(x_data)
W = tf.Variable(tf.random_uniform([1,len(x_data)], -5.0,5.0))

h = tf.matmul(W,x_data)

cost = tf.reduce_mean(tf.square(h-y_data))

a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(2000):
    sess.run(train)
    if step % 200 == 0:
        print step, sess.run(cost), sess.run(W)
