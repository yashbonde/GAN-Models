# importing the dependencies
import os # os interaction
import numpy as np # matrix math
import itertools # to iterate
import tensorflow as tf # ML
import matplotlib.pyplot as plt # visualisation
from tqdm import tqdm # progress bar
from tensorflow.examples.tutorials.mnist import input_data # training data
from tensorflow.python.framework.graph_util import convert_variables_to_constants # for frozen graph

# G(z)
def generator(z, batch_norm_training = False, reuse = False, debug = False):
	# this is the generator network, it takes the input the latent variable z and returns an image
	with tf.variable_scope('generator', reuse = reuse):
		# layer 1
		conv1 = tf.layers.conv2d_transpose(z, filters = 1024, kernel_size = (4, 4), strides = (1, 1), padding = 'VALID')
		lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training = batch_norm_training))
		
		# layer 2
		conv2 = tf.layers.conv2d_transpose(lrelu1, filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')
		lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training = batch_norm_training))

		# layer 3
		conv3 = tf.layers.conv2d_transpose(lrelu2, filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')
		lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training = batch_norm_training))
		
		# layer 4
		conv4 = tf.layers.conv2d_transpose(lrelu3, filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')
		lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training = batch_norm_training))

		# output layer, layer 5
		conv5 = tf.layers.conv2d_transpose(lrelu4, filters = 1, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')
		final_image = tf.tanh(conv5)

		if debug:
			print(lrelu1)
			print(lrelu2)
			print(lrelu3)
			print(lrelu4)
			print(final_image)

		return final_image

# D(x)
def discriminator(x, batch_norm_training = False, reuse = False, debug = False):
	# This is the discriminator model, this will be the exct opposite of generator network
	with tf.variable_scope('discriminator', reuse = reuse):
		# layer 1
		conv1 = tf.layers.conv2d(x, filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')
		lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv1, training = batch_norm_training))

		# layer 2
		conv2 = tf.layers.conv2d(lrelu1, filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')
		lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training = batch_norm_training))

		# layer 3
		conv3 = tf.layers.conv2d(lrelu2, filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')
		lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training = batch_norm_training))

		# layer 4
		conv4 = tf.layers.conv2d(lrelu3, filters = 1028, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')
		lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training = batch_norm_training))

		# output layer, layer 5
		conv5 = tf.layers.conv2d(lrelu4, filters = 1, kernel_size = (4, 4), strides = (1, 1), padding = 'VALID')
		final_ = tf.sigmoid(conv5)

		if debug:
			print(lrelu1)
			print(lrelu2)
			print(lrelu3)
			print(lrelu4)
			print(final_logits)

		return final_, conv5

def show_train_histogram(d_losses, g_losses, show = False, save = False, save_path = 'train_hist.png'):
	x = range(len(d_losses))
	# make the plot
	plt.plot(x, d_losses, label = 'D_loss')
	plt.plot(x, g_losses, label = 'G_loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(loc = 4)
	plt.grid(True)
	plt.tight_layout()
	# conditions
	if save:
		plt.savefig(save_path)
	if show:
		plt.show()
	else:
		plt.close()

def show_generated_images(some_fixed_z, num_epoch, show = False, save = False, save_path = None):
	# get the images
	test_images = sess.run(Gz, {z: some_fixed_z, batch_norm_training: False})

	# make the image
	side_len = 5
	figure, ax = plt.subplots(side_len, side_len, figsize = (5, 5))
	for i, j in itertools.product(range(side_len), range(side_len)):
		ax[i, j].get_xaxis().set_visible(False)
		ax[i, j].get_yaxis().set_visible(False)

	# put the images
	for k in range(side_len * side_len):
		i = k // side_len
		j = k % side_len
		ax[i, j].cla()
		ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap = 'gray')

	# make the labels
	label = 'Epoch {0}'.format(num_epoch)
	figure.text(0.5, 0.04, label, ha = 'center')

	# save
	if save:
		plt.savefig(save_path)
		print('Generated images stored at ' + save_path)

	# show
	if show:
		plt.show()
	else:
		plt.close()

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
	graph = session.graph
	with graph.as_default():
		freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
		output_names = output_names or []
		output_names += [v.op.name for v in tf.global_variables()]
		input_graph_def = graph.as_graph_def()
		if clear_devices:
			for node in input_graph_def.node:
				node.device = ""
		frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
	return frozen_graph

# constants
batch_size = 100
learning_rate = 0.0002
num_epochs = 20
some_fixed_z = np.random.normal(0, 1, (25, 1, 1, 100))

# now we make the placeholders
x = tf.placeholder(tf.float32, [None, 64, 64, 1], name = 'input_image_placeholder')
z = tf.placeholder(tf.float32, [None, 1, 1, 100], name = 'latent_placeholder')
batch_norm_training = tf.placeholder(tf.bool)

# making the network
print('[*]Making the network...')
Gz = generator(z, batch_norm_training)
Dreal, Dreal_logits = discriminator(x, batch_norm_training) # sigmoid, direct_conv
Dfake, Dfake_logits = discriminator(Gz, batch_norm_training, reuse = True) # sigmoid, direct_conv

# making the loss function
print('[*]Making loss functions...')
Dreal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dreal_logits, labels = tf.ones(shape = [batch_size, 1, 1, 1])))
Dfake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dfake_logits, labels = tf.zeros(shape = [batch_size, 1, 1, 1])))
D_loss = Dreal_loss + Dfake_loss
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dfake_logits, labels = tf.ones(shape = [batch_size, 1, 1, 1])))

# making the training function
trainable_vars = tf.trainable_variables()
D_vars = [v for v in trainable_vars if v.name.startswith('discriminator')]
G_vars = [v for v in trainable_vars if v.name.startswith('generator')]

# optmiser for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	discriminator_optimizer_step = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(D_loss, var_list = D_vars)
	generator_optmizer_step = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(G_loss, var_list = G_vars)

# initialising the session
print('[*]Initialising Session...')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# get the dataset
print('[*]Loading data...')
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True, reshape = [])
# images_all = tf.reshape(mnist.train.images, (len(images_all), 28, 28, 1))
images_all = tf.image.resize_images(mnist.train.images, size = (64, 64)).eval(session = sess)
images_all -= 0.5
images_all *= 2.

# results folder
root_dir = './MNIST_Example/'
if not os.path.isdir(root_dir):
	os.mkdir(root_dir)
images_dir = root_dir + 'images/'
if not os.path.isdir(images_dir):
	os.mkdir(images_dir)

# making the log dictionary
training_log = {'epoch': [], 'd_loss': [], 'g_loss': []}

# now training the network
for epoch in range(num_epochs):
	if epoch == 0:
		print('[*]Starting Training...')
	# for each training epoch
	d_losses = []
	g_losses = []
	range_len = (len(mnist.train.images) // batch_size) - 1
	for idx in tqdm(range(range_len)):
		# train for that particular batch, 1
		batch_images = images_all[idx*batch_size: (idx+1)*batch_size]

		# update the dicriminator
		# latent variable
		random_noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))
		feed_dict_d = {x:batch_images, z:random_noise, batch_norm_training:True}
		loss_d, _ = sess.run([D_loss, discriminator_optimizer_step], feed_dict_d)
		d_losses.append(loss_d)

		# update the generator
		# latent variable
		random_noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))
		feed_dict_g = {x: batch_images, z:random_noise, batch_norm_training:True}
		loss_g, _ = sess.run([G_loss, generator_optmizer_step], feed_dict_g)
		g_losses.append(loss_g)

	# update the log
	training_log['epoch'].append(epoch)
	training_log['d_loss'].append(np.mean(loss_d))
	training_log['g_loss'].append(np.mean(loss_g))

	# print the results
	print('[*]Epoch: {0}, d_loss: {1}, g_loss: {2}'.format(epoch, np.mean(loss_d), np.mean(loss_g)))

	# save the generated images
	save_path_gen_image = images_dir + 'gen_image' + str(epoch) + '.png'
	show_generated_images(some_fixed_z, epoch, save = True, save_path = save_path_gen_image)

	# save the new frozen graph after every few iterations
	if (epoch+1) % 5 == 0:
		show_train_histogram(training_log['d_loss'], training_log['g_loss'], save = True)		

		# now we need to save the model
		fg = freeze_session(session = sess)
		tf.train.write_graph(fg, root_dir, 'gan_mnist_simple'+str(epoch)+'.pb', as_text = False)

