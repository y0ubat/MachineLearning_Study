import tensorflow as tf

x_data = [1.,2.,3.,4.,5.]
y_data = [1.,2.,3.,4.,5.]

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

h = W*X

cost = tf.reduce_mean(tf.square(h-Y))

descent = W-tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W,X)-Y),X)))
update = W.assign(descent)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in xrange(10):
    sess.run(update,feed_dict={X:x_data,Y:y_data})
    print step, sess.run(cost,feed_dict={X:x_data,Y:y_data}), sess.run(W)

print sess.run(h,feed_dict={X:10})