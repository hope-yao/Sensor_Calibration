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

def run(ratio):
	input_data =np.load('./data/noisy_sensor_data.npy').item()
	test_input = input_data['test_input']
	test_output = input_data['test_output']
	train_input = input_data['train_input']
	train_output = input_data['train_output']

	idx = np.arange(500)
	np.random.shuffle(idx)
	train_input=train_input[idx]
	train_output=train_output[idx]

	train_input_max = []
	train_output_max = []
	test_input_max = []
	test_output_max = []

	for i, data_i in enumerate(train_input):
		# normalize with training data
		train_input_max += [np.max(data_i)]
		train_input[i] = data_i/np.max(data_i)
		data_i = train_output[i]
		train_output_max += [np.max(data_i)]
		train_output[i] = data_i/np.max(data_i)
	for i, data_i in enumerate(test_input):
		# normalize with testing data
		test_input_max += [np.max(data_i)]
		test_input[i] = data_i/np.max(data_i)
		data_i = test_output[i]
		test_output_max += [np.max(data_i)]
		test_output[i] = data_i/np.max(data_i)

	train_input_mag = np.expand_dims(np.asarray(train_input_max),1)
	train_output_mag = np.expand_dims(np.asarray(train_output_max),1)
	test_input_mag = np.expand_dims(np.asarray(test_input_max),1)
	test_output_mag = np.expand_dims(np.asarray(test_output_max),1)


	pca = PCA(n_components=500)
	pca.fit(train_input)
	train_input = pca.transform(train_input)
	test_input = pca.transform(test_input)


	batch_size = 16
	num_time_steps = 500
	input_pl = tf.placeholder(tf.float32, [batch_size, num_time_steps])
	output_pl = tf.placeholder(tf.float32, [batch_size, 3000])
	input_mag_pl = tf.placeholder(tf.float32, [batch_size, 1])
	output_mag_pl = tf.placeholder(tf.float32, [batch_size, 1])
	net = {}
	net['enc1'] = x = slim.fully_connected(input_pl, 512, scope='enc/fc1')
	net['enc2'] = x = slim.fully_connected(x, 512, scope='enc/fc2')
	net['enc3'] = x = slim.fully_connected(x, 256, scope='enc/fc3')
	net['enc4'] = x = slim.fully_connected(x, 256, scope='enc/fc4')
	net['enc5'] = x = slim.fully_connected(x, 128, scope='enc/fc5')
	net['enc6'] = x = slim.fully_connected(x, 128, scope='enc/fc6')
	z = x
	net['dec1'] = x = slim.fully_connected(x, 256, scope='dec/fc1')
	net['dec2'] = x = slim.fully_connected(x, 256, scope='dec/fc2')
	net['dec3'] = x = slim.fully_connected(x, 1024, scope='dec/fc3')
	net['dec4'] = x = slim.fully_connected(x, 1024, scope='dec/fc4')
	net['dec5'] = x = slim.fully_connected(x, 3000, scope='dec/fc5')
	net['dec6'] = x = slim.fully_connected(x, 3000, activation_fn=None, scope='dec/fc6')
	net['denoised'] = net['dec6'] #+ input_pl



	net['cls1'] = x = slim.fully_connected(z, 64, scope='ppn/fc1')
	net['cls11'] = x = slim.fully_connected(x, 64, scope='ppn/fc11')
	# extra = tf.tile(input_mag_pl, (1,32))
	# extra = tf.concat([net['cls1'], extra], 1)
	net['cls2'] = x = slim.fully_connected(x, 16, scope='ppn/fc2')
	net['cls21'] = x = slim.fully_connected(x, 16, scope='ppn/fc21')
	# extra = tf.tile(input_mag_pl, (1,8))
	# extra = tf.concat([net['cls2'], extra], 1)
	# extra = net['cls2']
	net['cls3'] = x = slim.fully_connected(x, 8, scope='ppn/fc3')
	x = tf.concat([x, input_mag_pl], 1)
	net['cls31'] = x = slim.fully_connected(x, 8, scope='ppn/fc31')
	net['cls32'] = x = slim.fully_connected(x, 8, scope='ppn/fc32')
	# extra = tf.tile(input_mag_pl, (1,2))
	# extra = tf.concat([net['cls3'], extra], 1)
	# extra = net['cls3']
	net['cls4'] = x = slim.fully_connected(x, 4, scope='ppn/fc4')
	net['cls4'] = x = slim.fully_connected(x, 2, activation_fn=None, scope='ppn/fc41')
	net['cls4'] = x = slim.fully_connected(x, 1, activation_fn=None, scope='ppn/fc42')
	net['mag'] = x+input_mag_pl

	loss_l2 = tf.reduce_mean(tf.abs(net['denoised'] - output_pl))
	loss_l1 = tf.Variable(0.)#tf.reduce_max(tf.abs(net['denoised'] - output_pl))#
	loss_mag = tf.reduce_mean(tf.abs( (net['mag'] - output_mag_pl)))
	loss_main = loss_l2 + loss_l1

	## OPTIMIZER ## note: both optimizer and learning rate is not found in the paper
	main_optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
	enc_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='enc')
	dec_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='dec')
	ppn_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='ppn')
	main_grads = main_optimizer.compute_gradients(loss_main, enc_vars+dec_vars)
	main_train_op = main_optimizer.apply_gradients(main_grads)
	ppn_optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.5)
	ppn_grads = ppn_optimizer.compute_gradients(loss_mag, ppn_vars)
	ppn_train_op = ppn_optimizer.apply_gradients(ppn_grads)
	train_op = tf.group(main_train_op, ppn_train_op)

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
	for eq_i in range(max_epoch):
		# training data, for optimization
		num_itr = train_input.shape[0] / batch_size
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
			denoised_data_val = sess.run(net['denoised'], {input_pl: input_data_val})
			denoised_mag_val = sess.run(net['mag'], {input_pl: input_data_val,
													 input_mag_pl: test_input_mag[i * batch_size:(i + 1) * batch_size]})
			denoised_data_val = denoised_data_val * denoised_mag_val
			reference_data_val = test_output[i * batch_size:(i + 1) * batch_size] * test_output_mag[
																					i * batch_size:(i + 1) * batch_size]
			er1 += [np.abs(denoised_mag_val - test_output_mag[i * batch_size:(i + 1) * batch_size]) / np.abs(
				test_output_mag[i * batch_size:(i + 1) * batch_size])]
			er2 += [np.sum(np.abs(denoised_data_val - reference_data_val), 1) / np.max(reference_data_val, 1)]
		print(np.mean(er1), np.mean(er2))

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




