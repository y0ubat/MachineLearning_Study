import tensorflow as tf


x_data = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
y_data = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

h = W*X + b
cost = tf.reduce_mean(tf.square(h-Y))

a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(20000):
    sess.run(train,feed_dict={X:x_data,Y:y_data})
    if step % 100 ==0:
        print step, sess.run(cost,feed_dict={X:x_data,Y:y_data}),sess.run(W), sess.run(b)


print sess.run(h,feed_dict={X:4.01})
