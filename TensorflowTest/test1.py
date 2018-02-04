import tensorflow as tf
import numpy as np


def add_layer(inputs,input_size,out_size,activation_function= None ):
    Weights = tf.Variable(tf.random_normal([input_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])+ 0.1)
    Wx_plus_b  =tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return  outputs

xs = tf.placeholder(dtype=tf.float32,shape =[None,1])

ys = tf.placeholder(dtype=tf.float32,shape =[None,1])
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise =np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)


loss =tf.reduce_mean(tf.reduce_sum( tf.square(ys-prediction),reduction_indices=[1]))
train_step  = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            # to see the step improvement
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


