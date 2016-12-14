
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
import os
import numpy as np

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
    def __init__(self, path_to_model=None):

	if not path_to_model is None:
	    self.load_model

        # some variables which are set by the training function
        self.img_shape = (32, 32, 1)
	self.kernel_shape = (5, 5)
        self.n_classes = 43
	
	# gradient descent parameters
	self.learning_rate = 0.00001
	self.batch_size = 20
	self.num_epochs = 50000
	self.dropout_prob = .5
	
	# some network properties
        self.n_kernels_c1 = 108
        self.n_kernels_c2 = 50
        self.n_neurons_d1 = 64
        self.pool_factor_1 = 4
        self.pool_factor_2 = 2


    def sanity_check(self, image_shape, label_shape):
	passed = True
	if self.img_shape != image_shape:
	    print "ERROR! Image shape does not correspond to predefined!"
	    passed = False
	
	if self.n_classes != label_shape[1]:
	    print "ERROR! Number of classes does not correspond to label shape!"
	    passed = False
	
	if image_shape[0] != label_shape[0]:
	    print "ERROR! Different number of training examples and labels!"
	    passed = False

	return passed


    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def load_model_and_evaluate(self, path, data):
	x = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]), name="images")
	keep_prob = tf.placeholder(tf.float32, name="dropout_prob")  # dropout (keep probability)

	prediction_op = self.feed_forward(x, keep_prob)
	batch_runs = data.shape[0] / self.batch_size
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    saver.restore(sess, path)
	    print "Model successfully loaded!"
	    
	    # evaluate the data on our pretrained model
	    predictions = np.zeros((data.shape[0], self.n_classes))
	    for batchIdx in range(batch_runs):
		local_predictions = sess.run(prediction_op,
						feed_dict={x: data[
							      batchIdx * self.batch_size:(batchIdx + 1) * self.batch_size],
							   keep_prob: 1.})
		predictions[batchIdx*self.batch_size:(batchIdx+1)*self.batch_size] = local_predictions
	
	    return predictions
    
    def feed_forward(self, images, keep_prob):
	# create variables for layer one
        W_conv1 = self.weight_variable([self.kernel_shape[0], self.kernel_shape[1], self.img_shape[2], self.n_kernels_c1], "kernels_layer1")
        b_conv1 = self.bias_variable([self.n_kernels_c1], "biases_layer1")
        
        # do convolution
        # and max pooling for layer one
        h_conv1 = tf.nn.relu(self.conv2d(images, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1, k=self.pool_factor_1)

        # initialize vars for layer two
        W_conv2 = self.weight_variable([self.kernel_shape[0], self.kernel_shape[1], self.n_kernels_c1, self.n_kernels_c2], "kernels_layer2")
        b_conv2 = self.bias_variable([self.n_kernels_c2], "biases_layer2")

        # convolve and max pool layer 2
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2, k=self.pool_factor_2)
        
        # now, do the dense layer 3
        reduced_img_w = self.img_shape[0] / (self.pool_factor_1*self.pool_factor_2)
        reduced_img_h = self.img_shape[1] / (self.pool_factor_1*self.pool_factor_2)
        
        W_fc1 = self.weight_variable([reduced_img_w * reduced_img_h * self.n_kernels_c2, self.n_neurons_d1], "weights_dense_layer3")
        b_fc1 = self.bias_variable([self.n_neurons_d1], "biases_dense_layer3")
        h_pool2_flat = tf.reshape(h_pool2, [-1, reduced_img_w*reduced_img_h*self.n_kernels_c2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # apply dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        # out_layer
        W_fc2 = self.weight_variable([self.n_neurons_d1, self.n_classes], "weights_out_layer4")
        b_fc2 = self.bias_variable([self.n_classes], "biases_out_layer4")
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	
	return y_conv

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
        
	y_conv = self.feed_forward(images, keep_prob)

        # add cross entropy as objective function
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, labels))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return train_step, loss, y_conv, accuracy

    def train_model(self, train_images, train_label, test_images, test_label, kernel_shape=(5, 5), learning_rate=0.001):
	
	# set some class variables before constructing the model
	if not self.sanity_check(train_images.shape, train_label.shape):
	    return -1

	train_size = train_images.shape[0]
	batch_runs = train_size / self.batch_size
	print "Batch size: " + str(self.batch_size)
	print "Number of iterations per epoch: " + str(batch_runs)

	# create the graph
	x = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]), name="images")
	y = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_classes), name="labels")
	keep_prob = tf.placeholder(tf.float32, name="dropout_prob")  # dropout (keep probability)
	train_op, ce_op, accuracy_op, prediction_op = self.construct_model(x, y, keep_prob)
	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	print "Graph successfully constructed! Start training..."
	
	accuracies_test = []
	accuracies_train = []
	cross_entropy_test = []
	with tf.Session() as sess:
	    sess.run(tf.initialize_all_variables())
	    with tf.device('/gpu:1'):
		for epoch in range(self.num_epochs):
		    for batchIdx in range(batch_runs):
			sess.run(train_op, feed_dict={x: train_images[batchIdx*self.batch_size:(batchIdx+1)*self.batch_size],
						      y: train_label[batchIdx*self.batch_size:(batchIdx+1)*self.batch_size],
						      keep_prob: self.dropout_prob})
			
			if batchIdx % (batch_runs / 2) == 0:  # the denominator determines the printing frequency (here twice every epoch)
			    total_acc_test = 0.
			    total_ce_test = 0.
			    total_acc_train = 0.
			    for test_batch in xrange(test_images.shape[0] / self.batch_size):
				total_acc_test += sess.run(accuracy_op,
							   feed_dict={x: test_images[
									 test_batch * self.batch_size:(test_batch + 1) * self.batch_size],
								      y: test_label[
									 test_batch * self.batch_size:(test_batch + 1) * self.batch_size],
								      keep_prob: 1.})
				total_ce_test += sess.run(ce_op,
							  feed_dict={x: test_images[
									test_batch * self.batch_size:(test_batch + 1) * self.batch_size],
								     y: test_label[
									test_batch * self.batch_size:(test_batch + 1) * self.batch_size],
								     keep_prob: 1.})
			    for train_batch in xrange(train_images.shape[0] / self.batch_size):
				total_acc_train += sess.run(accuracy_op,
							    feed_dict={x: train_images[
									train_batch * self.batch_size:(train_batch + 1) * self.batch_size],
								       y: train_label[
									train_batch * self.batch_size:(train_batch + 1) * self.batch_size],
								       keep_prob: 1.})

			    acc = total_acc_test / float(test_images.shape[0] / self.batch_size)
			    ce = total_ce_test / float(test_images.shape[0] / self.batch_size)
			    train_acc = total_acc_train / float(train_images.shape[0] / self.batch_size)
			    print "[Batch " + str(batchIdx) + "]\tAccuracy[Test]: " + str(acc) + "\tCross Entropy[Test]: " + str(ce) +\
				  "\tAccuracy[Train]: " + str(train_acc)
			    accuracies_test.append(acc)
			    accuracies_train.append(train_acc)
			    cross_entropy_test.append(ce)
		    print "Epoch " + str(epoch) + " done!"
	    
	    # save everything to directory
	    self.finish_after_training(saver, sess, accuracies_test, accuracies_train, cross_entropy_test)

    def evaluate_model(self, test_data):
	pass

    def finish_after_training(self, saver, sess, accuracies_test, accuracies_train, cross_entropy_test):
	
	print "Create Directory..."
	date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	os.mkdir('training/' + date_string)

	print "Save Model..."
	save_path = 'training/' + date_string + '/'
	path = saver.save(sess, save_path + 'model.ckpt')

	# save accuracies during training
	with open(save_path + 'accuracies.pkl', 'w') as f:
	    pickle.dump((accuracies_test, accuracies_train, cross_entropy_test), f, pickle.HIGHEST_PROTOCOL)
		
	print "Plot Accuracy..."
	fig = plt.figure(figsize=(14,8))
	plt.plot(accuracies_test, color='green', label='Accuracy on the test set')
	plt.plot(accuracies_train, color='red', label='Accuracy on the training set')
	plt.legend(loc="lower right")
	fig.savefig(save_path + 'plot.png', dpi=400)



if __name__ == "__main__":
    print "Loading the data..."
    with open('train_data_gray_norm_aug.pkl', 'rb') as train_handle:
	train_set, train_labels = pickle.load(train_handle)
    with open('test_data_gray_norm_aug.pkl', 'rb') as test_handle:
	test_set, test_labels = pickle.load(test_handle)
    print "Successfully loaded " + str(train_set.shape[0]) + " images!"

    classifier = cNN()
    classifier.train_model(train_set, train_labels, test_set, test_labels)
    print "Done with training!"
