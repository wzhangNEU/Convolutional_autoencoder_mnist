
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import math
import matplotlib
import sys
import matplotlib.pyplot as plt


# In[29]:


#some important variables
n_epochs = 101 #101
batch_size = 128
input_shape = [batch_size,28,28,1]
stride = [1,1]
learning_rate = 0.001
noise_factor = 0.3

#extract mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mean_img = np.mean(mnist.train.images, axis=0)


# In[30]:


def train_mnist():

	#tensorflow graph
	  #placeholders
	inputs = tf.placeholder(tf.float32, input_shape, name = 'inputs')
	targets = tf.placeholder(tf.float32, input_shape, name = 'targets')
	
	  #encoder
	conv1 = tf.layers.conv2d(inputs = inputs, filters = 4, kernel_size = (3,3), padding = 'SAME', strides = stride, activation= tf.nn.relu)
	maxpool1 = tf.layers.max_pooling2d(conv1, pool_size = (2,2), strides = (2,2), padding = 'SAME')
	conv2 = tf.layers.conv2d(inputs = maxpool1, filters = 4, kernel_size = (3,3), padding = 'SAME', strides = stride, activation = tf.nn.relu)
	maxpool2 = tf.layers.max_pooling2d(conv2, pool_size = (2,2), strides = (2,2), padding = 'SAME')
	conv3 = tf.layers.conv2d(inputs = maxpool2, filters = 8, kernel_size = (3,3), padding = 'SAME', strides = stride, activation = tf.nn.relu)
	maxpool3 = tf.layers.max_pooling2d(conv3, pool_size = (2,2), strides = (2,2), padding = 'SAME')
	#compressed internal representation
	  #decoder
	unpool1 = tf.image.resize_images(maxpool3, (7,7), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	deconv1 = tf.layers.conv2d_transpose(inputs = unpool1, filters = 8, kernel_size = (3,3), padding = 'SAME', strides = stride, activation = tf.nn.relu)
	unpool2 = tf.image.resize_images(deconv1, (14,14), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	deconv2 = tf.layers.conv2d_transpose(inputs = unpool2, filters = 4, kernel_size = (3,3), padding = 'SAME', strides = stride, activation = tf.nn.relu)
	unpool3 = tf.image.resize_images(deconv2, (28,28), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	deconv3 = tf.layers.conv2d_transpose(inputs = unpool3, filters = 4, kernel_size = (3,3), padding = 'SAME', strides = stride, activation = tf.nn.relu)
	
	output = tf.layers.dense(inputs=deconv3, units=1)
	output = tf.reshape(output, input_shape)
	
	
	 #performance metrics
	loss = tf.divide(tf.norm(tf.subtract(targets, output), ord = 'fro', axis = [1,2]), tf.norm(targets, ord = 'fro', axis = [1,2]))
	cost = tf.reduce_mean(loss)
	opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	
	#end of graph
	
	#figure for saving plot of reconstruction while training
	fig = plt.figure(1, figsize=(15,40))
	index = 1
	
	#training
	init_op = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init_op)
	n_batches = mnist.train.num_examples//batch_size
	
	
	for epoch in range(n_epochs):
		print '\nEpoch\t' + str(epoch + 1) + '/' + str(n_epochs)
		for batch in range(n_batches):
			batch_data,_ = mnist.train.next_batch(batch_size)
			training_batch = np.array([img - mean_img for img in batch_data])
			training_batch = np.reshape(training_batch, input_shape)
			noisy_batch = training_batch + noise_factor * np.random.randn(*training_batch.shape)
			batch_cost,_ = sess.run([cost, opt], feed_dict = {inputs: noisy_batch, targets: training_batch})
			if batch%1000 == 0:
				print batch_cost					
			print '\r' + str(((batch +1) * 100)/n_batches) + '%',
			sys.stdout.flush()
		print("Epoch: {}/{}...".format(epoch+1, n_epochs), "Training loss: {:.4f}".format(batch_cost))
		
		#Saving 2 sets of [a test image, input noisy image and reconstruction] for every 10th epoch along with first and fifth epoch.
		if epoch%10 == 0 or epoch == 5:
			batch_data, _ = mnist.test.next_batch(batch_size)
			testing_batch = np.array([img - mean_img for img in batch_data])
			testing_batch = np.reshape(testing_batch, input_shape)
			noisy_batch = testing_batch + noise_factor * np.random.randn(*testing_batch.shape)
			recon = sess.run(output, feed_dict={inputs: noisy_batch, targets: testing_batch})
			recon = np.array(recon)
			recon = np.reshape(recon, input_shape)
			#with pdf1.PdfPages('testImages.pdf') as pdf:
			for example_i in range(2):
				plt.subplot(12,6,index)
				index = index + 1
				plt.title('Epoch-' + str(epoch) + '-x')
				plt.imshow(np.reshape(testing_batch[example_i, :,:,0], (28, 28)), interpolation="nearest", cmap="gray")
				plt.grid(False)
				             
				plt.subplot(12,6,index)
				index = index + 1  
				plt.title('Epoch-' + str(epoch) + '-x`')   
				plt.imshow(np.reshape(noisy_batch[example_i, :,:,0], (28, 28)), interpolation="nearest", cmap="gray")
				plt.grid(False)
				
				plt.subplot(12,6,index)
				index = index + 1
				plt.title('Epoch-' + str(epoch) + '-r')
				plt.imshow(np.reshape(recon[example_i, :,:,0], (28, 28)), interpolation="nearest", cmap="gray")
				plt.grid(False)
				
	fig.savefig("output.pdf", bbox_inches='tight')
	
	#end of training
	
	# Saving activations
	fig2 = plt.figure(2, figsize=(10,27))
	index = 1
	
	batch_data, _ = mnist.test.next_batch(batch_size)
	testing_batch = np.array([img - mean_img for img in batch_data])
	testing_batch = np.reshape(testing_batch, input_shape)
	noisy_batch = testing_batch + noise_factor * np.random.randn(*testing_batch.shape)
	#retriving activations of Encoder and Decoder Layers; m3 is the compressed internal representation layer activations
	e1,e2,e3,m1,m2,m3 = sess.run([conv1,conv2,conv3,maxpool1,maxpool2,maxpool3], feed_dict={inputs: noisy_batch, targets: testing_batch})
	e1 = np.array(e1)
	e2 = np.array(e2)
	e3 = np.array(e3)
	m1 = np.array(m1)
	m2 = np.array(m2)
	m3 = np.array(m3)
	e1 = np.reshape(e1, [batch_size, 28,28,4])
	m1 = np.reshape(m1, [batch_size, 14,14,4])
	e2 = np.reshape(e2, [batch_size, 14,14,4])
	m2 = np.reshape(m2, [batch_size, 7,7,4])
	e3 = np.reshape(e3, [batch_size, 7,7,8])
	m3 = np.reshape(m3, [batch_size, 4,4,8])
	
	#saving filters of encoder1; we are selecting the first image of the batch as a stimuli to observe the activations
	for filter_i in range(4):
		plt.subplot(8,4,index)
		index = index + 1
		plt.title('Encoder1-' + 'Filter-' + str(filter_i+1))
		plt.imshow(np.reshape(e1[0, :,:,filter_i], (28, 28)), interpolation="nearest", cmap="gray")
		plt.grid(False)
	#saving filters of maxpool1;
	for filter_i in range(4):
		plt.subplot(8,4,index)
		index = index + 1
		plt.title('Maxpool1-' + 'Filter-' + str(filter_i+1))
		plt.imshow(np.reshape(m1[0, :,:,filter_i], (14, 14)), interpolation="nearest", cmap="gray")
		plt.grid(False)
	
	#saving filters of encoder2
	for filter_i in range(4):
		plt.subplot(8,4,index)
		index = index + 1
		plt.title('Encoder2-' + 'Filter-' + str(filter_i+1))
		plt.imshow(np.reshape(e2[0, :,:,filter_i], (14, 14)), interpolation="nearest", cmap="gray")
		plt.grid(False)
	#saving filters of maxpool2;
	for filter_i in range(4):
		plt.subplot(8,4,index)
		index = index + 1
		plt.title('Maxpool2-' + 'Filter-' + str(filter_i+1))
		plt.imshow(np.reshape(m2[0, :,:,filter_i], (7, 7)), interpolation="nearest", cmap="gray")
		plt.grid(False)
	
	#saving filters of encoder3
	for filter_i in range(8):
		plt.subplot(8,4,index)
		index = index + 1  
		plt.title('Encoder3-' + 'Filter-' + str(filter_i+1))   
		plt.imshow(np.reshape(e3[0, :,:,filter_i], (7, 7)), interpolation="nearest", cmap="gray")
		plt.grid(False)
	#saving filters of maxpool3;
	for filter_i in range(8):
		plt.subplot(8,4,index)
		index = index + 1
		plt.title('Maxpool3-' + 'Filter-' + str(filter_i+1))
		plt.imshow(np.reshape(m3[0, :,:,filter_i], (4, 4)), interpolation="nearest", cmap="gray")
		plt.grid(False)
	
	fig2.savefig("activations.pdf", bbox_inches='tight')
	
	
	sess.close()
	


if __name__ == '__main__':
	train_mnist()
			

