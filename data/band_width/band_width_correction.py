import numpy as np

def get_peak_width(x):
	peak_loc = np.argmax(x)
	peak_val = x[peak_loc]
	for i in range(0, peak_loc, 1):
		if x[i]> peak_val*0.1:
			st = i
			break
		if x[i] > peak_val * 0.05:
			st_ext = i
	for i in range(peak_loc, len(x), 1):
		if x[i]< peak_val*0.1:
			ed = i
		if x[i] < peak_val * 0.05:
			ed_ext = i
			break
	band_width = ed - st
	# sig = x[st_ext:ed_ext]
	sig = x[peak_loc-10:peak_loc+20]
	return peak_val, band_width, sig

def get_feature(data):
	input = data['input_data']
	output = data['output_data']
	pred = data['pred_data']

	pred_peak = []
	pred_band = []
	pred_sig = []
	output_peak = []
	output_band = []
	output_sig = []
	mag = []
	for i in range(len(pred)):
		# reference
		peak_val_i, band_width_i, sig_i = get_peak_width(output[i])
		output_peak += [peak_val_i]
		output_band += [[band_width_i]]
		mag_i = np.max(sig_i)
		mag += [mag_i]
		output_sig += [sig_i/mag_i]
		# calibrated
		peak_val_i, band_width_i, sig_i = get_peak_width(pred[i])
		pred_peak += [peak_val_i]
		pred_band += [[band_width_i]]
		pred_sig += [sig_i/mag_i]
	pred_peak = np.asarray(pred_peak,dtype='float32')
	pred_band = np.asarray(pred_band,dtype='float32')
	pred_sig = np.asarray(pred_sig,dtype='float32')
	output_peak = np.asarray(output_peak,dtype='float32')
	output_band = np.asarray(output_band,dtype='float32')
	output_sig = np.asarray(output_sig,dtype='float32')
	return pred_peak, pred_band, pred_sig, output_peak, output_band, output_sig

train_data = np.load('/home/hope-yao/Documents/Sensor_Calibration/data/band_width/calibrated_broken_train_data.npy').item()
test_data = np.load('/home/hope-yao/Documents/Sensor_Calibration/data/band_width/calibrated_broken_test_data.npy').item()
train_pred_peak, train_pred_band, train_pred_sig, train_output_peak, train_output_band, train_output_sig = get_feature(train_data)
test_pred_peak, test_pred_band, test_pred_sig, test_output_peak, test_output_band, test_output_sig = get_feature(test_data)

print('done')
import tensorflow as tf
import tensorflow.contrib.slim as slim
batch_size = 10
input_sig_pl = tf.placeholder(tf.float32, [None, 30], name='input_sig_pl')
input_width_pl = tf.placeholder(tf.float32, [None, 1], name='input_width_pl')
output_sig_pl = tf.placeholder(tf.float32, [None, 30], name='output_sig_pl')
output_width_pl = tf.placeholder(tf.float32, [None, 1], name='output_width_pl')
net = {}
net['enc1'] = x = slim.fully_connected(input_sig_pl, 32, activation_fn=None, scope='enc/fc1')
net['enc2'] = x = slim.fully_connected(x, 16, scope='enc/fc2')
net['enc3'] = x = slim.fully_connected(x, 8, scope='enc/fc3')
net['enc4'] = x = slim.fully_connected(x, 4, scope='enc/fc4')
x = tf.concat([x, input_width_pl],1)
net['enc5'] = x = slim.fully_connected(x, 3, scope='enc/fc5')
net['band_res'] = x = slim.fully_connected(x, 1, activation_fn=None, scope='enc/fc6')
net['band'] = net['band_res'] + input_width_pl

# net['enc1_peak'] = x = slim.fully_connected(input_sig_pl, 32, activation_fn=None, scope='enc_peak/fc1')
# net['enc2_peak'] = x = slim.fully_connected(x, 16, scope='enc_peak/fc2')
# net['enc3_peak'] = x = slim.fully_connected(x, 8, scope='enc_peak/fc3')
# net['enc4_peak'] = x = slim.fully_connected(x, 4, scope='enc_peak/fc4')
# x = tf.concat([x, input_width_pl],1)
# net['enc5_peak'] = x = slim.fully_connected(x, 3, scope='enc_peak/fc5')
# net['band_peak'] = x = slim.fully_connected(x, 1, activation_fn=None, scope='enc_peak/fc6')

loss_main = tf.reduce_mean(tf.abs(net['band']-output_width_pl))
main_optimizer = tf.train.AdamOptimizer(1e-2, beta1=0.5)
main_grads = main_optimizer.compute_gradients(loss_main)
main_train_op = main_optimizer.apply_gradients(main_grads)
# ppn_optimizer = tf.train.AdamOptimizer(1e-4, beta1=0.5)
# ppn_grads = ppn_optimizer.compute_gradients(loss_mag, ppn_vars)
# ppn_train_op = ppn_optimizer.apply_gradients(ppn_grads)
# train_op = tf.group(main_train_op, ppn_train_op)




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

for eq_i in range(max_epoch):
	# training data, for optimization
	train_loss = []
	test_loss = []

	# testing data
	num_itr = 16#test_input.shape[0] / batch_size
	for i in range(1,num_itr,1):
		feed_dict_train = {input_sig_pl: test_pred_sig[i*batch_size:(i+1)*batch_size],
		                   input_width_pl: test_pred_band[i*batch_size:(i+1)*batch_size],
		                   output_sig_pl: test_output_sig[i*batch_size:(i+1)*batch_size],
		                   output_width_pl: test_output_band[i*batch_size:(i+1)*batch_size],
		                   }
		test_loss_i = sess.run(loss_main, feed_dict_train)
		test_loss += [test_loss_i]

	num_itr = 50#train_input.shape[0] / batch_size
	for i in range(num_itr):
		feed_dict_train = {input_sig_pl: train_pred_sig[i*batch_size:(i+1)*batch_size],
		                   input_width_pl: train_pred_band[i*batch_size:(i+1)*batch_size],
		                   output_sig_pl: train_output_sig[i*batch_size:(i+1)*batch_size],
		                   output_width_pl: train_output_band[i*batch_size:(i+1)*batch_size],
		                   }

		train_loss_i, _ = sess.run([loss_main, main_train_op], feed_dict_train)
		train_loss += [train_loss_i]

	print(eq_i, np.mean(train_loss), np.mean(test_loss))

