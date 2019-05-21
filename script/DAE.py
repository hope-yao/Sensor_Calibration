import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def run(ratio):

	input_data =np.load('./data/noisy_sensor_data.npy').item()
	test_input = input_data['test_input']
	test_output = input_data['test_output']
	train_input = input_data['train_input']
	train_output = input_data['train_output']

	batch_size = 16
	num_time_steps = 3000
	input_pl = tf.placeholder(tf.float32, [batch_size, num_time_steps])
	output_pl = tf.placeholder(tf.float32, [batch_size, num_time_steps])
	net = {}
	net['enc1'] = slim.fully_connected(input_pl, 1024, scope='enc/fc1')
	net['enc2'] = slim.fully_connected(net['enc1'], 1024, scope='enc/fc2')
	net['enc3'] = slim.fully_connected(net['enc2'], 512, scope='enc/fc3')
	net['enc4'] = slim.fully_connected(net['enc3'], 512, scope='enc/fc4')
	net['enc5'] = slim.fully_connected(net['enc4'], 256, scope='enc/fc5')
	net['enc6'] = slim.fully_connected(net['enc5'], 256, scope='enc/fc6')
	net['dec1'] = slim.fully_connected(net['enc6'], 512, scope='dec/fc1')
	net['dec2'] = slim.fully_connected(net['dec1'], 512, scope='dec/fc2')
	net['dec3'] = slim.fully_connected(net['dec2'], 1024, scope='dec/fc3')
	net['dec4'] = slim.fully_connected(net['dec3'], 1024, scope='dec/fc4')
	net['dec5'] = slim.fully_connected(net['dec4'], 3000, scope='dec/fc5')
	net['dec6'] = slim.fully_connected(net['dec5'], 3000, activation_fn=tf.identity, scope='dec/fc6')
	net['residual'] = net['dec6']
	net['denoised'] = net['residual'] #+ input_pl
	loss_l2 = tf.reduce_mean(tf.abs(net['denoised'] - output_pl))
	loss_l1 = tf.reduce_max(tf.abs(net['denoised'] - output_pl))
	loss = loss_l2 + loss_l1 #max error regularizer

	## OPTIMIZER ## note: both optimizer and learning rate is not found in the paper
	optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.5)
	grads = optimizer.compute_gradients(loss, tf.all_variables())
	train_op = optimizer.apply_gradients(grads)

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
	print(train_input.shape, test_input.shape)
	for eq_i in range(max_epoch):
		# training data, for optimization
		num_itr = train_input.shape[0] / batch_size
		ave_loss_l1_val_train = []
		ave_loss_l2_val_train = []
		for i in range(num_itr):
			feed_dict_train = {input_pl: train_input[i*batch_size:(i+1)*batch_size],
						 output_pl: train_output[i*batch_size:(i+1)*batch_size]}
			loss_l2_val, loss_l1_val, _ = sess.run([loss_l2, loss_l1, train_op], feed_dict_train)
			ave_loss_l1_val_train += [loss_l1_val]
			ave_loss_l2_val_train += [loss_l2_val]
		train_loss_l1_val_hist += [np.mean(ave_loss_l1_val_train)]
		train_loss_l2_val_hist += [np.mean(ave_loss_l2_val_train)]
		# testing data
		ave_loss_l1_val_test = []
		ave_loss_l2_val_test = []
		num_itr = test_input.shape[0] / batch_size
		for i in range(num_itr):
			feed_dict_test = {input_pl: test_input[i*batch_size:(i+1)*batch_size],
						 output_pl: test_output[i*batch_size:(i+1)*batch_size]}
			loss_l2_val, loss_l1_val = sess.run([loss_l2, loss_l1], feed_dict_test)
			ave_loss_l2_val_test += [loss_l2_val]
			ave_loss_l1_val_test += [loss_l1_val]
		test_loss_l1_val_hist += [np.mean(ave_loss_l1_val_test)]
		test_loss_l2_val_hist += [np.mean(ave_loss_l2_val_test)]

		print(eq_i, np.mean(ave_loss_l1_val_train), np.mean(ave_loss_l1_val_test), np.mean(ave_loss_l2_val_train), np.mean(ave_loss_l2_val_test))

		er1 = []
		er2 = []
		for j in range(10):
			input_data_val = test_input[j * batch_size:(j + 1) * batch_size]
			denoised_data_val = sess.run(net['denoised'], {input_pl: input_data_val})
			reference_data_val = test_output[j * batch_size:(j + 1) * batch_size]
			er1 += [np.abs(np.max(denoised_data_val - reference_data_val, 1)) / np.max(reference_data_val, 1)]
			er2 += [np.sum(np.abs(denoised_data_val - reference_data_val), 1) / np.max(reference_data_val, 1)]
		print(np.mean(er1), np.mean(er2))


	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(train_loss_l2_val_hist[3:], label='training l2 loss')
	plt.plot(test_loss_l2_val_hist[3:], label='testing l2 loss')
	plt.legend()
	plt.subplot(2,1,2)
	plt.plot(train_loss_l1_val_hist[3:], label='training l1 loss')
	plt.plot(test_loss_l1_val_hist[3:], label='testing l1 loss')
	plt.legend()
	plt.show()

	#plt.plot(sess.run(net['output'], feed_dict_train)[idx], label='true')
	i = 0
	input_data_val = test_input[i*batch_size:(i+1)*batch_size]
	residual_data_val = sess.run(net['residual'], {input_pl: input_data_val})
	denoised_data_val = sess.run(net['denoised'], {input_pl: input_data_val})
	reference_data_val = test_output[i*batch_size:(i+1)*batch_size]


if __name__ == '__main__':
	# for ratio in range(0.,1,10):
	run(0.5)



