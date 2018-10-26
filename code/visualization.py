import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ch1=np.load('/home/hope-yao/Desktop/Sensor_Calibration/ch1_data.npy')
ch1_peak=np.max(ch1,1)[:,0]
sns.distplot(ch1_peak, label='ch1')

ch2=np.load('/home/hope-yao/Desktop/Sensor_Calibration/ch2_data.npy')
ch2_peak=np.max(ch2,1)[:,0]
sns.distplot(ch2_peak, label='ch2')

ch3=np.load('/home/hope-yao/Desktop/Sensor_Calibration/ch3_data.npy')
ch3_peak=np.max(ch3,1)[:,0]
sns.distplot(ch3_peak, label='ch3')

plt.legend()




plt.plot(ch1[0])
plt.plot(ch2[0])
plt.plot(ch1[0])

noisy_data_1 = np.load('ch1_data.npy')[:, :, 0]  # noisy signal, totally off
noisy_data_2 = np.load('ch2_data.npy')[:, :, 0]  # noisy signal
# noisy_data = np.concatenate([noisy_data_1[:330], noisy_data_2[330:]], 0)
noisy_data = noisy_data_2
good_data = np.load('ch3_data.npy')[:, :, 0]  # good signal
plt.subplot(3,1,1)
plt.imshow(np.log(abs(noisy_data_1))/np.max(np.log(abs(noisy_data_1))),cmap='Reds')
plt.axis('off')
plt.subplot(3,1,2)
plt.imshow(np.log(abs(noisy_data_2))/np.max(np.log(abs(noisy_data_2))),cmap='Reds')
plt.axis('off')
plt.subplot(3,1,3)
plt.imshow(np.log(abs(good_data))/np.max(np.log(abs(good_data))),cmap='Reds')
plt.axis('off')

np.load('input_data_val_ch2.npy')
np.load('output_data_val_ch2.npy')
np.load('denoised_data_val_ch2.npy')

