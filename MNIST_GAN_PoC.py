# This file is just to establish a Proof of Concept

# import the dependencies
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# import the data set
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# looking as some of the images
plt.imshow(np.reshape(train_data[123], [28, 28]))
print(train_labels[123])

# We need to reshape all the image files and combine it into one dataset
images_ = np.vstack([train_data, eval_data])
images_ = np.asarray([np.reshape(i, [28,28,1]) for i in images_])
print(images_.shape)

# we combine the labels too
labels_ = np.hstack([train_labels, eval_labels])
print(labels_.shape)

# make discriminator network
def _make_discriminator(input_layer):
    """
    This is a normal convolution network to predict either 0 or 1. The following
    architecture is the one used on tensorflow site to as tutorial to convoluti-
    -onal networks.
    """
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    
    # Logits Layer
    # 2 units -> 1 for true or 0 for false
    logits = tf.layers.dense(inputs=dropout, units=2)
    
    return logits

# make generator network
def _make_generator(input_layer):
    """
    This is a bit more tricky to implement as it uses deconvolution blocks to m-
    -ake an image. I will try to mimick the architecture of convolutional blocks.
    """
    # dense layer #1
    dense1 = tf.layers.dense(inputs=input_layer, units = 1024, activation = tf.nn.relu)
    dropout = tf.layers.dropout(inputs = dense1, rate = 0.4)
    
    # dense layer #2
    dense2 = tf.layers.dense(inputs = dropout, units = 7*7*64, activation = tf.nn.relu)
    dense2_ = tf.reshape(dense2, [-1, 7, 7, 64])
    dense_upsample = tf.image.resize_images(dense2_, [14,14])

    # convolution layer
    conv1_ = tf.layers.conv2d_transpose(
        inputs = dense_upsample,
        filters = 32,
        kernel_size = [5,5],
        padding = "same",
        activation = tf.nn.relu
        )
    conv1_upsample = tf.image.resize_images(conv1_, [24, 24])

    # convolution layer #2
    conv2_ = tf.layers.conv2d_transpose(
        inputs = conv1_upsample,
        filters = 1,
        kernel_size = [5,5],
        activation = tf.nn.sigmoid)
    return conv2_

# make batch generators
def get_noise_batch(batch_size):
    I = np.identity(10, dtype = np.float32)
    batch = np.asarray([np.random.randint(10) for _ in range(batch_size)])
    return I[batch]

def get_real_batch(x, batch_size):
    ids = np.arange(len(x))
    np.random.shuffle(ids)
    batch = ids[:batch_size]
    return x[batch]

# define hyper-paramters
num_epochs = 100000
batch_size = 256
disp_step = 100

# make the complete model

# define the placeholders
# I will be defining 3 placeholders, this just makes for better understanding of code
# for loop k
p_gen_1 = tf.placeholder(tf.float32, [batch_size, 10]) # line 3 in pseudocode
p_dis_real = tf.placeholder(tf.float32, [batch_size, 28, 28, 1]) # line 4 pseudocode
# after k loop
p_gen_2 = tf.placeholder(tf.float32, [batch_size, 10]) # line 7 pseudocode

# getting values for the loss function
# for loop k
pred_gen_1 = _make_discriminator(_make_generator(p_gen_1)) # D(G(z))
pred_real = _make_discriminator(p_dis_real) # D(x)
# after k loop
pred_gen_2 = _make_discriminator(_make_generator(p_gen_2)) # D(G(z))

# define the loss functions
loss_d = tf.reduce_sum(tf.log(pred_real) + tf.log(1 - pred_gen_1))/batch_size
loss_g = tf.reduce_sum(tf.log(1 - pred_gen_2))/batch_size

# train steps
opt = tf.train.AdamOptimizer()
train_step_d = opt.minimize(loss_d)
train_step_g = opt.minimize(loss_g)

# train the model
loss_d_list = []
loss_g_list = []
for e in range(num_epochs):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # train discriminator
    feed_dict_d = {p_gen_1: get_noise_batch(batch_size),
                  p_dis_real: get_real_batch(images_, batch_size),
                  p_gen_2: get_noise_batch(batch_size)}
    _, l_d = sess.run([train_step_d, loss_d], feed_dict = feed_dict_d)
    loss_d_list.append(l_d)
    
    # train generator
    feed_dict_g = {p_gen_2: get_noise_batch(batch_size),
                  p_gen_1: get_noise_batch(batch_size),
                  p_dis_real: get_real_batch(images_, batch_size)}
    _, l_g = sess.run([train_step_g, loss_g], feed_dict = feed_dict_g)
    loss_g_list.append(l_g)
    
    if (e+1) % disp_step == 0:
        feed_dict_disp = {p_gen_1: get_noise_batch(batch_size),
                  p_dis_real: get_real_batch(images_, batch_size),
                         p_gen_2: get_noise_batch(batch_size)}
        l_d, l_g = sess.run([loss_d, loss_g], feed_dict = feed_dict_disp)
        print("Epoch: {0}; Discriminator Loss: {1}, Generator Loss: {2}".format(e, l_d, l_g))

        
