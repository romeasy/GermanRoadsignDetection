import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
This class implements a convolutional neural network classifier.
Main usage should consist of two steps, namely train and evaluate. During training, the weights of the network
will be changed in a way to nicely represent the data and classify it in the end.

Parameters:
    img_shape:
        Shape of the images that will be presented to the
        network as (width, height, #channels)
    
    learning_rate:
        The learning rate used for gradient descent
    
    architecture:
        List. It contains for each layer the number
        of neurons. For the convolution, the number
        of neurons corresponds to the number of kernels.
"""
class cNN:
    
    def __init__(self, architecture, img_shape=(28,28), kernel_shape=(5,5), learning_rate=0.001):
        if len(architecture) > 5:
            print "ERROR. The network is too deep. So far, we can't deal with more than 5 layers!"
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.kernel_shape = kernel_shape
        
        # some variables which are set by the training function
        self.img_shape = img_shape #(x, y, channels)
        self.n_classes = 10
        
    
    """
    This function generates lists of weight matrices and bias matrices from
    some given architecture.
    These can then be used to construct the network.
    """
    def generate_weights_and_biases(self):
        weights = []
        biases = []
        for layer in xrange(self.architecture):
            if self.architecture[layer][0] == "conv":
                if layer == 0: # first layer
                    last_output = img_shape[2]
                else:
                    last_output = self.architecture[layer-1][1]
                weights.append(tf.Variable(tf.random_normal([self.kernel_shape[0],
                                                            self.kernel_shape[1],
                                                            img_shape[2],
                                                            self.architecture[layer][1]]
                                                           )))
                biases.append(tf.Variable(tf.random_normal([self.architecture[layer][1]])))

            elif self.architecture[layer][0] == "dense":
                last_output = self.architecture[layer-1][1]
                num_of_conv_layers_so_far = len([i for i in xrange(layer) if self.architecture[i] == "conv"])
                num_of_input_units = (img_shape[0] * img_shape[1]) / num_of_conv_layers_so_far**2
                weights.append(tf.Variable(tf.random_normal([num_of_input_units, self.architecture[layer][1]])))
                biases.append(tf.Variable(tf.random_normal([self.architecture[layer][1]])))
            
            elif self.architecture[layer][0] == "out":
                last_output = self.architecture[layer-1][1]
                weights.append(tf.Variable(tf.random_normal([last_output, self.architecture[layer][1]])))
                biases.append(tf.Variable(tf.random_normal([self.architecture[layer][1]])))
                
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    """
    This function creates the graph for training and returns a tensorflow function object.
    This object can then be used to train the batches.
    
    Parameters:
        images:
            A batch of images with shape: (n_batch, width, height, n_channels)
        labels:
            A batch of labels encoded as one-hot vector with shape:
            (n_batch, n_classes)
        keep_prob:
            A number that indicates the dropout probability
    """
    def construct_model(self, images, labels, keep_prob):
        
        # some network properties
        n_kernels_c1 = 50
        n_kernels_c2 = 64
        n_neurons_d1 = 1024
        pool_factor_1 = 2
        pool_factor_2 = 2
        
        images = tf.reshape(images, shape=[-1, 28, 28, 1])
        # create variables for layer one
        W_conv1 = self.weight_variable([kernel_shape[0], kernel_shape[1], self.img_shape[2], n_kernels_c1])
        b_conv1 = self.bias_variable([n_kernels_c1])
        
        # do convolution
        # and max pooling for layer one
        h_conv1 = tf.nn.relu(self.conv2d(images, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1, k=pool_factor_1)

        # initialize vars for layer two
        W_conv2 = self.weight_variable([kernel_shape[0], kernel_shape[1], n_kernels_c1, n_kernels_c2])
        b_conv2 = self.bias_variable([n_kernels_c2])

        # convolve and max pool layer 2
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2, k=pool_factor_2)
        
        # now, do the dense layer 3
        reduced_img_w = self.img_shape[0] / (pool_factor_1*pool_factor_2)
        reduced_img_h = self.img_shape[1] / (pool_factor_1*pool_factor_2)
        
        W_fc1 = self.weight_variable([reduced_img_w * reduced_img_h * n_kernels_c2, n_neurons_d1])
        b_fc1 = self.bias_variable([n_neurons_d1])
        h_pool2_flat = tf.reshape(h_pool2, [-1, reduced_img_w*reduced_img_h*n_kernels_c2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # apply dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        # out_layer
        W_fc2 = self.weight_variable([n_neurons_d1, self.n_classes])
        b_fc2 = self.bias_variable([self.n_classes])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        # add cross entropy as objective function
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return (train_step, cross_entropy, accuracy)

    def train_model(self, mnist):
        
        
        # set some class variables before constructing the model
        self.n_classes = 10
        self.img_shape = (28, 28, 1)
        batch_size = 10
        train_size = 10000
        batch_runs = train_size / batch_size
        print "Batch size: " + str(batch_size)

        # create the graph
        x = tf.placeholder(tf.float32, shape=(batch_size, np.prod(self.img_shape)))
        y = tf.placeholder(tf.float32, shape=(batch_size, self.n_classes))
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        train_op, ce_op, accuracy_op = self.construct_model(x, y, keep_prob)
        print "Graph successfully constructed! Start training..."

        self.accuracies = []
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for batchIdx in range(batch_runs):
                    batch_img, batch_labels = mnist.train.next_batch(batch_size)
                    sess.run(train_op, feed_dict={x: batch_img,
                                                  y: batch_labels,
                                                  keep_prob: 1.})
                    
                    if batchIdx % (batch_runs / 10) == 0:
                        acc = sess.run(accuracy_op, feed_dict={x: batch_img,
                                                               y: batch_labels,
                                                               keep_prob: 1.})
                        ce = sess.run(ce_op, feed_dict={x: batch_img,
                                                        y: batch_labels,
                                                        keep_prob: 1.})
                        print "[Batch " + str(batchIdx) + "]\tAccuracy: " + str(acc) + "\tCross Entropy: " + str(ce)
                        self.accuracies.append(acc)
                        
                print "Epoch " + str(epoch) + " done!"

        fig = plt.figure(figsize=(14, 8))
        plt.xlabel('Training time')
        plt.ylabel('Accuracy')
        plt.plot(np.array(self.accuracies))
        plt.savefig('test.png')



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
architecture = [("conv", 32), ("conv", 64), ("dense", 1024), ("out", 43)]
kernel_shape = (5, 5)
learning_rate = 0.001
classifier = cNN(architecture)
classifier.train_model(mnist)
