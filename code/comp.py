import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

regr = linear_model.LinearRegression()
aa=np.load('./data/noisy_sensor_data.npy').item()
regr.fit(aa['train_input'], aa['train_output'])
test_output = regr.predict(aa['test_input'])

peak_err = np.mean(np.abs(np.max(aa['test_output'],1) - np.max(test_output,1))/np.max(aa['test_output'],1))
err = np.abs(aa['test_output'] - test_output)
overall_err = np.mean(np.sum(err / np.max(test_output,1, keepdims=True),1))


# TOO MANY FEATURES, SVM WONT WORK
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)
test_output = svr_rbf.fit(aa['train_input'], aa['train_output']).predict(aa['test_input'])


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
#X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=3000, max_features=20)
regr.fit(aa['train_input'], aa['train_output'])
print(regr.feature_importances_)
test_output = regr.predict(aa['test_input'])

