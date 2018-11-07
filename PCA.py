import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


input_data = np.load('./data/noisy_sensor_data.npy', encoding='latin1').item()
test_input = input_data['test_input']
test_output = input_data['test_output']
train_input = input_data['train_input']
train_output = input_data['train_output']


pca = PCA(n_components=500)
pca.fit(train_input)
PCA(copy=True, n_components=500, whiten=False)

cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumvar)
plt.show()
