
# coding: utf-8
# Author:   Roman Schulte-Sasse
# Mail:     r.schulte-sasse@fu-berlin.de
# Date:     5th of june, 2016

# # Training a cNN to detect roadsigns
# In order to process roadsigns in the autonomous car of the Freie Universität, we want to train a
# convolutional (deep) neural network.
# 
# This network is supposed to distinguish between different classes of signs (stop, attention, train crossing etc)
# and the final model will then be integrated to the autonomos ROS structure.
# This file contains the implementation of the cNN.
# It can be run locally, though I recommend using a nice and fast GPU for that.
#
# Cuda and CuDNN should be available for efficient GPU training with tensorflow.
#
# Note further that this code relies on the data that is preprocessed by a notebook called 'preprocessing.ipynb'


import tensorflow as tf
import pickle
import time
from datetime import datetime
import matplotlib.pyplot as plt

class cNN:
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
    def __init__(self, _architecture, _img_shape=(28, 28), _kernel_shape=(5, 5), _learning_rate=0.001):
        if len(architecture) > 5:
            print "ERROR. The network is too deep. So far, we can't deal with more than 5 layers!"
        self.learning_rate = _learning_rate
        self.architecture = _architecture
        self.kernel_shape = _kernel_shape
        
        # some variables which are set by the training function
        self.img_shape = _img_shape  # (x, y, channels)
        self.n_classes = 10

    """
    This function generates lists of weight matrices and bias matrices from
    some given architecture.
    These can then be used to construct the network.
    """
    def generate_weights_and_biases(self):
        weights = []
        biases = []
        for layer in xrange(len(self.architecture)):
            if self.architecture[layer][0] == "conv":
                if layer == 0:  # first layer
                    last_output = self.img_shape[2]
                else:
                    last_output = self.architecture[layer-1][1]
                weights.append(tf.Variable(tf.random_normal([self.kernel_shape[0],
                                                            self.kernel_shape[1],
                                                            last_output,
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
            	assert layer == (len(architecture) - 1) # sanity check
                last_output = self.architecture[layer-1][1]
                weights.append(tf.Variable(tf.random_normal([last_output, self.architecture[layer][1]])))
                biases.append(tf.Variable(tf.random_normal([self.architecture[layer][1]])))
            else:
            	print "Unknown layer type! " + self.architecture[layer][0]
        return weights, biases
                
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

	def construct_graph(self, images, labels, weights, biases, keep_prob):
		last_output = images
		for layer in xrange(len(self.architecture)):
			local_weight = weights[layer]
			local_bias = biases[layer]
			if self.architecture[layer][0] == "conv":
				convoluted = tf.nn.relu(self.conv2d(last_output, local_weights) + local_bias)
				pooled = self.max_pool_2x2(convoluted, k=4) # FIXME: k arbitrary
				current_output = pooled # for next layer
			elif self.architecture[layer][0] == "dense":
				dense_out = tf.nn.relu(tf.matmul(current_output, local_weight) + local_bias)
			elif self.architecture[layer][0] == "out":
				pass
			else:
				print "Unknown layer type! " + self.architecture[layer][0]
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
        n_kernels_c1 = 108
        n_kernels_c2 = 50
        n_neurons_d1 = 64
        pool_factor_1 = 4
        pool_factor_2 = 2

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
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, labels))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return train_step, loss, accuracy

    def train_model(self, train_images, train_label, test_images, test_label):
        
        with tf.device('/gpu:0'):
		    print train_images.shape
		    assert(train_images.shape[0] == train_label.shape[0])
		    
		    # set some class variables before constructing the model
		    self.n_classes = train_label.shape[1]
		    self.img_shape = train_images[0].shape
		    train_size = train_images.shape[0]
		    batch_size = 20
		    batch_runs = train_size / batch_size
		    print "Batch size: " + str(batch_size)
		    print "Number of iterations per epoch: " + str(batch_runs)

		    # create the graph
		    x = tf.placeholder(tf.float32, shape=(batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]))
		    y = tf.placeholder(tf.float32, shape=(batch_size, self.n_classes))
		    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
		    train_op, ce_op, accuracy_op = self.construct_model(x, y, keep_prob)
		    print "Graph successfully constructed! Start training..."
		    
		    accuracies_test = []
		    accuracies_train = []
		    cross_entropy_test = []
		    with tf.Session() as sess:
		        sess.run(tf.initialize_all_variables())
		        for epoch in range(300):
		            for batchIdx in range(batch_runs):
		                sess.run(train_op, feed_dict={x: train_images[batchIdx*batch_size:(batchIdx+1)*batch_size],
		                                              y: train_label[batchIdx*batch_size:(batchIdx+1)*batch_size],
		                                              keep_prob: 1.})
		                
		                if batchIdx % (batch_runs / 2) == 0:  # the denominator determines the printing frequency (here twice every epoch)
		                    total_acc_test = 0.
		                    total_ce_test = 0.
		                    total_acc_train = 0.
		                    for test_batch in xrange(test_images.shape[0] / batch_size):
		                        total_acc_test += sess.run(accuracy_op,
		                                                   feed_dict={x: test_images[
		                                                                 test_batch * batch_size:(test_batch + 1) * batch_size],
		                                                              y: test_label[
		                                                                 test_batch * batch_size:(test_batch + 1) * batch_size],
		                                                              keep_prob: 1.})
		                        total_ce_test += sess.run(ce_op,
		                                                  feed_dict={x: test_images[
		                                                                test_batch * batch_size:(test_batch + 1) * batch_size],
		                                                             y: test_label[
		                                                                test_batch * batch_size:(test_batch + 1) * batch_size],
		                                                             keep_prob: 1.})
		                    for train_batch in xrange(train_images.shape[0] / batch_size):
		                        total_acc_train += sess.run(accuracy_op,
		                                                    feed_dict={x: train_images[
		                                                                train_batch * batch_size:(train_batch + 1) * batch_size],
		                                                               y: train_label[
		                                                                train_batch * batch_size:(train_batch + 1) * batch_size],
		                                                               keep_prob: 1.})

		                    acc = total_acc_test / float(test_images.shape[0] / batch_size)
		                    ce = total_ce_test / float(test_images.shape[0] / batch_size)
		                    train_acc = total_acc_train / float(train_images.shape[0] / batch_size)
		                    print "[Batch " + str(batchIdx) + "]\tAccuracy[Test]: " + str(acc) + "\tCross Entropy[Test]: " + str(ce) +\
		                          "\tAccuracy[Train]: " + str(train_acc)
		                    accuracies_test.append(acc)
		                    accuracies_train.append(train_acc)
		                    cross_entropy_test.append(ce)
		            print "Epoch " + str(epoch) + " done!"

		print "Save model..."
		date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
		with open(date_string + '_accuracies.pkl', 'w') as f:
			pickle.dump((accuracies_test, accuracies_train, cross_entropy_test), f, pickle.HIGHEST_PROTOCOL)
		
		print "Plot accuracy..."
		fig = plt.figure(figsize=(14,8))
		plt.plot(accuracies_test, color='green', label='Accuracy on the test set')
		plt.plot(accuracies_train, color='red', label='Accuracy on the training set')
		plt.legend(loc="lower right")
		fig.savefig(date_string + '_plot.png', dpi=400)

print "Loading the data..."
with open('train_data_norm.pkl', 'rb') as train_handle:
    train_set, train_labels = pickle.load(train_handle)
with open('test_data_norm.pkl', 'rb') as test_handle:
    test_set, test_labels = pickle.load(test_handle)
print "Successfully loaded " + str(train_set.shape[0]) + " images!"

architecture = [("conv", 32), ("conv", 64), ("dense", 1024), ("out", 43)]
img_shape = train_set[0].shape
print "Images have shape: " + str(img_shape)
kernel_shape = (5, 5)
learning_rate = 0.00001
classifier = cNN(architecture, img_shape, kernel_shape, learning_rate)
classifier.train_model(train_set, train_labels, test_set, test_labels)
print "Done with training!"
