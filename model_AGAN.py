import tensorflow as tf # ML
import numpy as np # matrix maths
from six.moves import xrange # saves memory

class AGAN(object):
	"""
	DocString for AGAN
	Args:
		session: a tensorflow session to run the model
		total_audio: a list of numpy array having the total audio in numbers, padded to proper length
		total_text: a list or numpy array having the integer text padded to a proper length
		seq_length: length of the audio after being padded
		text_length: length of the text after being padded
		n_epochs: total number of epochs to train
		learning_rate: learning rate of the model
		d_step: number of steps after which to display the stats of model
		save_step: number of steps after which to save the model in checkpoint file
	"""
	def __init__(self, session = None, total_audio = None, total_text = None, seq_length = 630850,
		text_length = 178, n_epochs = 100, learning_rate = 0.03, d_step = 10, save_step = 50,
		saver = None, verbose = True):
		# defining the audio and text input
		self.data_audio = total_audio
		self.data_gen = total_text

		# defining the placeholders
		self.sess = session
		self.text_len = text_length
		self.seq_len = seq_length
		self.verbose = verbose
		self.audio_placeholder = tf.placeholder(tf.float32, shape = [None, self.seq_len, 1], name = 'audio_placeholder')
		self.text_placeholder = tf.placeholder(tf.float32, shape = [None, self.text_len], name ='text_placeholder')

		# defining the hyper parameters
		self.n_epochs = n_epochs
		self.learning_rate = learning_rate
		self.display_step  = d_step
		self.save_step = save_step

	def _discriminator(self, input_audio):
		'''
		This function when called will give proabability of the audio either being generated or
			real. With time it will learn to discriminate better.
		'''
		# ____First Layer (Convolution)____
		self.d_w1 = tf.Variable(tf.truncated_normal(shape = [1000, 1, 16]), name = 'd_w1')
		self.d_b1 = tf.Variable(tf.truncated_normal(shape = [16]), name = 'd_b1')
		d1 = tf.nn.conv1d(input_audio, filters = self.d_w1, stride = 100, padding = 'SAME') + self.d_b1
		if self.verbose:
			print('[*]d1:',d1)

		# ____Second Layer (Convolution)____
		self.d_w2 = tf.Variable(tf.truncated_normal(shape = [200, 16, 32]), name = 'd_w2')
		self.d_b2 = tf.Variable(tf.truncated_normal(shape = [32]), name = 'd_b2')
		d2 = tf.nn.conv1d(d1, filters = self.d_w2, stride = 40, padding = 'SAME') + self.d_b2
		if self.verbose:
			print('[*]d2:',d2)

		# ____Third Layer (Convolution)____
		self.d_w3 = tf.Variable(tf.truncated_normal(shape = [100, 32, 128]), name = 'd_w3')
		self.d_b3 = tf.Variable(tf.truncated_normal(shape = [128]), name = 'd_b3')
		d3 = tf.nn.conv1d(d2, filters = self.d_w3, stride = 20, padding = 'SAME') + self.d_b3
		if self.verbose:
			print('[*]d3:',d3)

		# ____Fourth Layer (Dense)____
		d3 = tf.reshape(d3, [-1, 8*128])
		self.d_w4 = tf.Variable(tf.truncated_normal(shape = [8*128, 1024]), name = 'd_w4')
		self.d_b4 = tf.Variable(tf.truncated_normal(shape = [1024]), name = 'd_b4')
		d4 = tf.nn.relu(tf.matmul(d3, self.d_w4) + self.d_b4)
		if self.verbose:
			print('[*]d4:',d4)

		# ____Fifth Layer (Dense)____
		self.d_w5 = tf.Variable(tf.truncated_normal(shape = [1024, 1]), name = 'd_w5')
		self.d_b5 = tf.Variable(tf.truncated_normal(shape = [1]), name = 'd_b5')
		d5 = tf.matmul(d4, self.d_w5) + self.d_b5
		if self.verbose:
			print('[*]d5:',d5)

		return d5

	def _generator(self, input_text):
		'''
		When called this function will generate an audio sample from the given text
		'''
		# ____First Layer (Dense)____
		self.g_w1 = tf.Variable(tf.truncated_normal(shape = [self.text_len, 1380]), name = 'g_w1')
		self.g_b1 = tf.Variable(tf.truncated_normal(shape = [1380]), name = 'g_b1')
		g1 = tf.matmul(input_text, self.g_w1) + self.g_b1
		g1 = tf.contrib.layers.batch_norm(g1)
		g1 = tf.reshape(g1, [-1, 1380, 1])
		if self.verbose:
			print('[*]g1:',g1)

		# ____Second Layer (Convolution)____
		self.g_w2 = tf.Variable(tf.truncated_normal(shape = [400, 1, 512]), name = 'g_w2')
		self.g_b2 = tf.Variable(tf.truncated_normal(shape = [512]), name = 'g_b2')
		g2 = tf.nn.conv1d(g1, filters = self.g_w2, stride = 40, padding = 'SAME') + self.g_b2
		g2 = tf.contrib.layers.batch_norm(g2)
		g2 = tf.reshape(g2, [-1, g2.get_shape().as_list()[-1]*g2.get_shape().as_list()[-2], 1])
		if self.verbose:
			print('[*]g2:',g2)


		# ____Third Layer (Convolution)____
		self.g_w3 = tf.Variable(tf.truncated_normal(shape = [200, 1, 256]), name = 'g_w3')
		self.g_b3 = tf.Variable(tf.truncated_normal(shape = [256]), name = 'g_b3')
		g3 = tf.nn.conv1d(g2, filters = self.g_w3, stride = 40, padding = 'SAME') + self.g_b3
		g3 = tf.reshape(g3, [-1, g3.get_shape().as_list()[-1]*g3.get_shape().as_list()[-2], 1])
		if self.verbose:
			print('[*]g3:',g3)

		# ____Fourth Layer (Convolution)____
		self.g_w4 = tf.Variable(tf.truncated_normal(shape = [200, 1, 110]), name = 'g_w3')
		self.g_b4 = tf.Variable(tf.truncated_normal(shape = [110]), name = 'g_b3')
		g4 = tf.nn.conv1d(g3, filters = self.g_w4, stride = 20, padding = 'SAME') + self.g_b4
		g4 = tf.reshape(g4, [-1, g4.get_shape().as_list()[-1]*g4.get_shape().as_list()[-2], 1])
		if self.verbose:
			print('[*]g4:',g4)

		return g4

	def build_model(self):
		'''
		This is the first module that should be called once the model is imported into the application
		'''
		generated_audio = self._generator(self.text_placeholder) # it holds the generated audio
		if self.verbose:
			print(generated_audio)
		dx = self._discriminator(self.audio_placeholder) # it holds the probabilities for real audio
		self.verbose = False # making verbose false once all teh details of teh model are given.
		dg = self._discriminator(generated_audio) # it holds the probabilities for fake audio

		# Since generator wants that all the generated audio to fool the discriminator, so the loss
		# will be calculated between dg and 1. We will be using the tf.nn.sigmoid_cross_entropy_with_logits,
		# which calculates sigmoid value over an unscaled range.
		self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dg, labels = tf.ones_like(dg)))

		# Now from prespective of the discriminator, its goal is to get correct labels (1 for actual audio, 0 for generated audio).
		# So we would like to calculate loss for both the fake and real audio.
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dx, labels = tf.ones_like(dx)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dg, labels = tf.zeros_like(dg)))
		d_loss = self.d_loss_real + self.d_loss_fake

		# now writing the updating part
		t_vars = tf.trainable_variables()

		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]

		with tf.variable_scope(tf.get_variable_scope(), reuse = None) as scope:
			# We will be using the Adam optimiser
			self.d_trainer_fake = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss_fake, var_list = d_vars)
			self.d_trainer_real = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss_real, var_list = d_vars)
			self.g_trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.generator_loss, var_list = g_vars)

		print('[!]Model Build, Ready for training...')

	def train(self):
		'''
		Doc String for train
		'''
		# defining loss
		g_loss = 0.0
		dloss_fake, dloss_real = 0.0, 0.0
		d_real_count, d_fake_count, g_count = 0, 0, 0

		for e in xrange(self.n_epochs):
			self.sess.run(tf.global_variables_initializer())

			feed_dict = {self.text_placeholder: self.data_gen, self.audio_placeholder: self.data_audio}

			# requirements as to when to train the model
			if dloss_fake > 0.6:
				_, dloss_real, dloss_fake, g_loss = self.sess.run([self.d_trainer_fake, self.d_loss_real, self.d_loss_fake, self.generator_loss],
					feed_dict = feed_dict)
				d_fake_count += 1

			if g_loss > 0.5:
				_, dloss_real, dloss_fake, g_loss = self.sess.run([self.g_trainer, self.d_loss_real, self.d_loss_fake, self.generator_loss],
					feed_dict = feed_dict)
				g_count += 1

			if dloss_real > 0.45:
				_, dloss_real, dloss_fake, g_loss = self.sess.run([self.d_trainer_real, self.d_loss_real, self.d_loss_fake, self.generator_loss], 
					feed_dict = feed_dict)
				d_real_count += 1

			# display and saving the model
			if (e+1) % self.display_step == 0:
				dloss_real, dloss_fake, g_loss = self.sess.run([self.d_loss_real, self.d_loss_fake, self.generator_loss], feed_dict = feed_dict)
				print('[*]epoch:{0}; dloss_real:{1}; dloss_fake:{2}; g_loss:{3}'.format(e+1, dloss_real, dloss_fake, g_loss))
			'''
			if (e+1) % self.save_step == 0:
				save_path = self.saver.save(self.sess, "models/pretrained_gan.ckpt", global_step = e)
				print("saved to",save_path)
			'''

	def test(self):
		pass
		