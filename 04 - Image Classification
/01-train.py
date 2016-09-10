import tensorflow as tf
import input_data
import random
import matplotlib.pyplot as plt

path = './DATA/'
checkpoint_dir = "./cps/"


training_epochs = 25
batch_size = 100

data = input_data.read_data_sets(path,one_hot=True)


X = tf.placeholder('float',[None,784])
Y = tf.placeholder('float',[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(X,W)+b)

learning_rate = 0.1
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:

    saver = tf.train.Saver()

    if tf.gfile.Exists(checkpoint_dir + '/model.ckpt'):
        saver.restore(sess, checkpoint_dir + '/model.ckpt')
    else:
        init = tf.initialize_all_variables()
        sess.run(init)


    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(data.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = data.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost += sess.run(cost, feed_dict={X:batch_xs,Y:batch_ys})/total_batch

        if epoch % 1  == 0:
            print "Epoch :", (epoch+1), "cost =", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    r = random.randint(0,data.test.num_examples-1)

    print "Label: ", sess.run(tf.argmax(data.test.labels[r:r+1],1))
    print "Prediction: ", sess.run(tf.argmax(activation,1), {X:data.test.images[r:r+1]})

    correct_prediction = tf.equal(tf.argmax(activation,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print "Accuracy:", accuracy.eval({X:data.test.images,Y:data.test.labels})

    saver.save(sess, checkpoint_dir + 'model.ckpt')

    plt.imshow(data.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

