import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.decomposition import PCA

def model(input_pl, input_mag_pl, scope):
	net = {}
	with tf.variable_scope('enc'):
		with tf.variable_scope(scope):
			x = slim.fully_connected(input_pl, 1024)
			x = slim.fully_connected(x, 256)
			x = slim.fully_connected(x, 128)
			x = slim.fully_connected(x, 64)
			x = slim.fully_connected(x, 32)
			z = x = slim.fully_connected(x, 32)

	with tf.variable_scope('dec') :
		with tf.variable_scope(scope) :
			x = tf.layers.dense(x, 256)
			x = tf.layers.dense(x, 256)
			x = tf.layers.dense(x, 512)
			x = tf.layers.dense(x, 1024)
			x = tf.layers.dense(x, 3000)
			x = tf.layers.dense(x, 3000, activation=None)

	net['shape'] = tf.squeeze(x)[:, :3000]

	with tf.variable_scope('ppn'):
		with tf.variable_scope(scope):
			x = slim.fully_connected(z, 16)
			x = slim.fully_connected(x, 16)
			x = slim.fully_connected(x, 8)
			x = slim.fully_connected(x, 8)
			x = slim.fully_connected(x, 8)
			x = tf.concat([x, input_mag_pl], 1)
			x = slim.fully_connected(x, 8)
			x = slim.fully_connected(x, 8)
			x = slim.fully_connected(x, 4)
			x = slim.fully_connected(x, 2)
			x = slim.fully_connected(x, 1)
			net['mag'] = x + input_mag_pl

	net['denoised'] = net['shape'] * net['mag']
	return net

def build_rnn(input_pl, input_mag_pl, output_pl, output_mag_pl):
	net = {}
	with tf.variable_scope(tf.get_variable_scope()) as scope:
		sig_shape, sig_mag = input_pl, input_mag_pl
		for i in range(3):
			my_scope = 'rnn_{}'.format(i)
			net_i = model(sig_shape, sig_mag, my_scope)
			# tf.get_variable_scope().reuse_variables()

			loss_l2 = tf.reduce_mean(tf.abs(net_i['shape'] - output_pl))
			loss_l1 = tf.reduce_max(tf.abs(net_i['shape'] - output_pl))  # tf.Variable(0.)#
			net_i['loss_mag'] = loss_mag = tf.reduce_mean(tf.abs((net_i['mag'] - output_mag_pl)))
			net_i['loss_main'] = loss_main = loss_l2 + loss_l1
			enc_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='enc/{}'.format(my_scope))
			dec_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='dec/{}'.format(my_scope))
			ppn_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ppn/{}'.format(my_scope))
			## OPTIMIZER ## note: both optimizer and learning rate is not found in the paper
			main_optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
			main_grads = main_optimizer.compute_gradients(loss_main, enc_vars + dec_vars)
			main_train_op = main_optimizer.apply_gradients(main_grads)
			ppn_optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
			ppn_grads = ppn_optimizer.compute_gradients(loss_mag, ppn_vars)
			ppn_train_op = ppn_optimizer.apply_gradients(ppn_grads)
			net_i['train_op'] = train_op = tf.group(main_train_op, ppn_train_op)

			net[my_scope] = net_i
			sig_shape, sig_mag = net_i['shape'], net_i['mag']

	return net

def load_data():
	input_data = np.load('./data/broken_sensor_data.npy').item()
	test_input = input_data['test_input']
	test_output = input_data['test_output']
	train_input = input_data['train_input']
	train_output = input_data['train_output']

	idx = np.arange(500)
	np.random.shuffle(idx)
	train_input = train_input[idx]
	train_output = train_output[idx]
#train_data_dict, test_data_dict

	train_input_mag = []
	train_output_max = []
	test_input_max = []
	test_output_max = []

	for i, data_i in enumerate(train_input):
		# normalize with training data
		train_input_mag += [np.max(data_i)]
		train_input[i] = data_i / np.max(data_i)
		data_i = train_output[i]
		train_output_max += [np.max(data_i)]
		train_output[i] = data_i / np.max(data_i)
	for i, data_i in enumerate(test_input):
		# normalize with testing data
		test_input_max += [np.max(data_i)]
		test_input[i] = data_i / np.max(data_i)
		data_i = test_output[i]
		test_output_max += [np.max(data_i)]
		test_output[i] = data_i / np.max(data_i)

	train_input_mag = np.expand_dims(np.asarray(train_input_mag), 1)
	train_output_mag = np.expand_dims(np.asarray(train_output_max), 1)
	test_input_mag = np.expand_dims(np.asarray(test_input_max), 1)
	test_output_mag = np.expand_dims(np.asarray(test_output_max), 1)
    #train_input, train_output, test_input, test_output, train_input_mag, train_output_mag, test_input_mag, test_output_mag



def pregressive_train_rnn():
    num_itr = train_input.shape[0] // batch_size
    ave_loss_l1_val_train = []
    ave_loss_l2_val_train = []
    ave_loss_mag_val_train = []
    for i in range(num_itr):
        #np.save('train_input_mag',train_input_mag[i*batch_size:(i+1)*batch_size])
        #np.save('train_output_mag',train_output_mag[i*batch_size:(i+1)*batch_size])
        #exit(0)
        feed_dict_train = {input_pl: train_input[i*batch_size:(i+1)*batch_size],
                     output_pl: train_output[i*batch_size:(i+1)*batch_size],
                     input_mag_pl: train_input_mag[i*batch_size:(i+1)*batch_size],
                     output_mag_pl: train_output_mag[i*batch_size:(i+1)*batch_size],
                      }
        loss_l2_val, loss_l1_val, loss_mag_val, _ = sess.run([loss_l2, loss_l1, loss_mag, train_op], feed_dict_train)
        ave_loss_l1_val_train += [loss_l1_val]
        ave_loss_l2_val_train += [loss_l2_val]
        ave_loss_mag_val_train += [loss_mag_val]
    train_loss_l1_val_hist += [np.mean(ave_loss_l1_val_train)]
    train_loss_l2_val_hist += [np.mean(ave_loss_l2_val_train)]
    train_loss_mag_val_hist += [np.mean(ave_loss_mag_val_train)]
    # testing data
    ave_loss_l1_val_test = []
    ave_loss_l2_val_test = []
    ave_loss_mag_val_test = []
    num_itr = test_input.shape[0] / batch_size
    for i in range(num_itr):
        feed_dict_test = {input_pl: test_input[i*batch_size:(i+1)*batch_size],
                     output_pl: test_output[i*batch_size:(i+1)*batch_size],
                     input_mag_pl: test_input_mag[i*batch_size:(i+1)*batch_size],
                     output_mag_pl: test_output_mag[i*batch_size:(i+1)*batch_size],
                      }
        loss_l2_val, loss_l1_val, loss_mag_val = sess.run([loss_l2, loss_l1, loss_mag], feed_dict_test)
        ave_loss_l2_val_test += [loss_l2_val]
        ave_loss_l1_val_test += [loss_l1_val]
        ave_loss_mag_val_test += [loss_mag_val]
    test_loss_l1_val_hist += [np.mean(ave_loss_l1_val_test)]
    test_loss_l2_val_hist += [np.mean(ave_loss_l2_val_test)]
    test_loss_mag_val_hist += [np.mean(ave_loss_mag_val_test)]

    print(eq_i, np.mean(ave_loss_l1_val_train), np.mean(ave_loss_l1_val_test), np.mean(ave_loss_l2_val_train), np.mean(ave_loss_l2_val_test), np.mean(ave_loss_mag_val_train), np.mean(ave_loss_mag_val_test))

    er1 = []
    er2 = []
    for i in range(10):
        input_data_val = test_input[i * batch_size:(i + 1) * batch_size]
        denoised_data_val = sess.run(net_i['shape'], {input_pl: input_data_val})
        denoised_mag_val = sess.run(net_i['mag'], {input_pl: input_data_val,
                                                 input_mag_pl: test_input_mag[i * batch_size:(i + 1) * batch_size]})
        denoised_data_val = denoised_data_val * denoised_mag_val
        reference_data_val = test_output[i * batch_size:(i + 1) * batch_size] * test_output_mag[
                                                                                i * batch_size:(i + 1) * batch_size]
        er1 += [np.abs(denoised_mag_val - test_output_mag[i * batch_size:(i + 1) * batch_size]) / np.abs(
            test_output_mag[i * batch_size:(i + 1) * batch_size])]
        er2 += [np.sum(np.abs(denoised_data_val - reference_data_val), 1) / np.max(reference_data_val, 1)]
    print(np.mean(er1), np.mean(er2))

def run(ratio):
	train_data_dict, test_data_dict = load_data()

	batch_size = 16
	num_time_steps = 3000
	input_pl = tf.placeholder(tf.float32, [batch_size, num_time_steps])
	output_pl = tf.placeholder(tf.float32, [batch_size, 3000])
	input_mag_pl = tf.placeholder(tf.float32, [batch_size, 1])
	output_mag_pl = tf.placeholder(tf.float32, [batch_size, 1])

	net = build_rnn(input_pl, input_mag_pl, output_pl, output_mag_pl)

	## training starts ###
	FLAGS = tf.app.flags.FLAGS
	tfconfig = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement=True,
	)
	tfconfig.gpu_options.allow_growth = True
	sess = tf.Session(config=tfconfig)
	init = tf.global_variables_initializer()
	sess.run(init)


	max_epoch = 2000
	train_loss_l1_val_hist = []
	test_loss_l1_val_hist = []
	train_loss_l2_val_hist = []
	test_loss_l2_val_hist = []
	train_loss_mag_val_hist = []
	test_loss_mag_val_hist = []

	print(train_input.shape, test_input.shape)
	for net_i in net:
		progressive_train_rnn()

def visualize():
	plt.figure()
	plt.subplot(3,1,1)
	plt.plot(train_loss_l2_val_hist[3:], label='training l2 loss')
	plt.plot(test_loss_l2_val_hist[3:], label='testing l2 loss')
	plt.legend()
	plt.subplot(3,1,2)
	plt.plot(train_loss_l1_val_hist[3:], label='training l1 loss')
	plt.plot(test_loss_l1_val_hist[3:], label='testing l1 loss')
	plt.legend()
	plt.show()
	plt.subplot(3,1,3)
	plt.plot(train_loss_mag_val_hist[3:], label='training mag loss')
	plt.plot(test_loss_mag_val_hist[3:], label='testing mag loss')
	plt.legend()
	plt.show()

	#plt.plot(sess.run(net['output'], feed_dict_train)[idx], label='true')
	i = 0
	input_data_val = test_input[i*batch_size:(i+1)*batch_size]
	input_mag_val = test_input_mag[i*batch_size:(i+1)*batch_size]
	residual_data_val = sess.run(net['denoised'], {input_pl: input_data_val})
	denoised_data_val = sess.run(net['denoised'], {input_pl: input_data_val})
	denoised_mag_val = sess.run(net['mag'], {input_pl: input_data_val, input_mag_pl: input_mag_val})
	reference_data_val = test_output[i*batch_size:(i+1)*batch_size]

	if 1:
		test_max_val = test_input_mag[i*batch_size:(i+1)*batch_size,0:1]
		test_min_val = test_input_mag[i*batch_size:(i+1)*batch_size,1:]
		input_data_val = (input_data_val) * (test_max_val - test_min_val)  + test_min_val
		test_max_val = denoised_mag_val[:,0:1]
		test_min_val = denoised_mag_val[:,1:]
		residual_data_val = (residual_data_val) * (test_max_val - test_min_val)  + test_min_val
		denoised_data_val = (denoised_data_val) * (test_max_val - test_min_val)  + test_min_val
		test_max_val = test_output_mag[i*batch_size:(i+1)*batch_size,0:1]
		test_min_val = test_output_mag[i*batch_size:(i+1)*batch_size,1:]
		reference_data_val = (reference_data_val) * (test_max_val - test_min_val)  + test_min_val

	np.save('input_data_val_ch2.npy', input_data_val)
	np.save('output_data_val_ch2.npy', reference_data_val)
	np.save('denoised_data_val_ch2.npy', denoised_data_val)

	for idx in range(5):
		plt.figure(figsize=(7,15))
		plt.subplot(5,1,1)
		plt.plot(input_data_val[idx],label='bad sensor')
		plt.legend()
		plt.subplot(5,1,2)
		plt.plot(residual_data_val[idx], label='residual sensor')
		plt.legend()
		plt.subplot(5,1,3)
		plt.plot(denoised_data_val[idx], label='denoised signal')
		plt.legend()
		plt.subplot(5,1,4)
		plt.plot(reference_data_val[idx], label='good sensor')
		plt.legend()
		plt.subplot(5,1,5)
		plt.plot(reference_data_val[idx]-denoised_data_val[idx], label='error')
		plt.legend()
	plt.show()

if __name__ == '__main__':
	# for ratio in range(0.,1,10):
	run(0.5)




