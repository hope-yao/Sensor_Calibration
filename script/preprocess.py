import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

data_dir = './Data'
ch1_data = []
ch2_data = []
ch3_data = []
for dir_i in os.listdir(data_dir):
	if dir_i[-8:] == 'filtered':
		ch1_dir = os.path.join(data_dir, dir_i, 'Chanel1')
		ch2_dir = os.path.join(data_dir, dir_i, 'Chanel2')
		ch3_dir = os.path.join(data_dir, dir_i, 'Chanel3')
		for fn_i in os.listdir(ch1_dir):
			ch1_fn = os.path.join(ch1_dir, fn_i)
			ch2_fn = os.path.join(ch2_dir, fn_i)
			ch3_fn = os.path.join(ch3_dir, fn_i)
			if os.path.exists(ch2_fn) and os.path.exists(ch3_fn):
				ch1_data_i = sio.loadmat(ch1_fn)['data1_fil_maw']
				ch2_data_i = sio.loadmat(ch2_fn)['data2_fil_maw']
				ch3_data_i = sio.loadmat(ch3_fn)['data3_fil_maw']
				ch1_data += [ch1_data_i]
				ch2_data += [ch2_data_i]
				ch3_data += [ch3_data_i]
				if 0:
					plt.figure()
					plt.plot(ch2_data, label='ch3')
					plt.plot(ch3_data, label='ch2')
					plt.legend()
					plt.show()
			else:
				print('error in {} {}'.format(dir_ij, fn_i))


np.save('ch2_data.npy', ch2_data)
np.save('ch3_data.npy', ch3_data)
np.save('ch1_data.npy', ch1_data)



ch1 = np.load('ch1_data.npy')
ch2 = np.load('ch2_data.npy')
ch3 = np.load('ch3_data.npy')
idx = np.arange(660)
np.random.shuffle(idx)
ch1_test = np.squeeze(ch1[idx[:160]])
ch2_test = np.squeeze(ch2[idx[:160]])
ch3_test = np.squeeze(ch3[idx[:160]])
ch1_train = np.squeeze(ch1[idx[160:]])
ch2_train = np.squeeze(ch2[idx[160:]])
ch3_train = np.squeeze(ch3[idx[160:]])
broken_sensor_data = {'train_input': ch1_train, 'train_output': ch3_train, 'test_input': ch1_test, 'test_output': ch3_test}
np.save('broken_sensor_data', broken_sensor_data)
noisy_sensor_data = {'train_input': ch2_train, 'train_output': ch3_train, 'test_input': ch2_test, 'test_output': ch3_test}
np.save('noisy_sensor_data', noisy_sensor_data)

aa=np.load('noisy_sensor_data.npy').item()
er1 = np.mean(np.abs(np.max(aa['train_output'],1)-np.max(aa['train_input'],1))/np.max(aa['train_input'],1))
er2 = np.mean(np.abs(np.max(aa['test_output'],1)-np.max(aa['test_input'],1))/np.max(aa['test_input'],1))
print(er1, er2)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
HUGE_SIZE = 20
plt.rc('font', size=HUGE_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=HUGE_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=HUGE_SIZE)    # legend fontsize
plt.rc('figure', titlesize=HUGE_SIZE)  # fontsize of the figure title

from matplotlib.ticker import FormatStrFormatter
ax = plt.figure(figsize=(5,5))

# ax = plt.subplot(1,3,3)
sns.distplot(np.max(ch3_train,1), label='sensor A', hist=False)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

# ax = plt.subplot(1,3,2)
sns.distplot(np.max(ch1_train,1), label='sensor B', hist=False)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

# ax = plt.subplot(1,3,1)
sns.distplot(np.max(ch2_train,1), label='sensor C', hist=False)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

plt.xlabel('Peak accceleration (g)')
plt.legend()