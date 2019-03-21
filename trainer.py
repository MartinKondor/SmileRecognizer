import os
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.utils.validation import check_random_state
from sklearn.externals import joblib
from utils import load_all_img_data, load_targets, read_out_img 


check_random_state(50)
files = os.listdir('data')
data = []

# reading in data from files and resize them to the same size
for file in files:
    data.append(np.array(Image.fromarray(np.loadtxt('data/' + file, delimiter='\t'))\
			.resize((300, 168), Image.ANTIALIAS).getdata()))

X = np.array(data)
y = load_targets('data')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


def test_model(model, model_name):
	global X_train, X_test, y_train, y_test	
	print(model_name + ' error on train:', mse(y_train, model.predict(X_train)))
	print(model_name + ' error on test:', mse(y_test, model.predict(X_test)))
	print()

# estimators
"""
from sklearn.neural_network import MLPClassifier
print('Started training the MLPClassifier ...')
nn_model = MLPClassifier(
	hidden_layer_sizes=(250,),
 	activation='relu',
	solver='adam',
	alpha=0.0001,
	batch_size='auto',
	learning_rate='constant',
	learning_rate_init=0.0011,
	power_t=0.5,
	max_iter=1000,
	shuffle=True,
	tol=0.001,
	warm_start=False,
	momentum=1.,
	nesterovs_momentum=True,
	early_stopping=False,
	validation_fraction=0.1,
	beta_1=0.9,
	beta_2=0.999,
	epsilon=1e-08,
	random_state=0,
)\
.fit(X_train, y_train)
test_model(nn_model, 'MLPClassifier')
joblib.dump(nn_model, 'trained_models/mlpnn_model.pkl')
"""

from sklearn.svm import SVC
print('Started training the SVC ...')
svc_model = SVC(
	C=100.0,
	kernel='linear',
	degree=20,
	gamma=0.001,
	coef0=1.,
	shrinking=True,
	probability=True,
	tol=0.01,
	cache_size=200,
	class_weight='balanced', 
	max_iter=-1,
	decision_function_shape='ovr',
)\
.fit(X_train, y_train)
test_model(svc_model, 'SVC')
joblib.dump(svc_model, 'trained_models/svc_model.pkl')
exit()

from sklearn.neighbors import KNeighborsClassifier
print('Started training the KNN ...')
knn_model = KNeighborsClassifier(
	n_neighbors=4,
	weights='uniform',
	algorithm='auto',
	leaf_size=20,
	p=2,
	metric='minkowski',
	n_jobs=-1
)\
.fit(X_train, y_train)
test_model(knn_model, 'kNN')
joblib.dump(svc_model, 'trained_models/knn_model.pkl')


from sklearn.ensemble import GradientBoostingClassifier
print('Started training the GBC ...')
gbc_model = GradientBoostingClassifier(
	loss='deviance',
	learning_rate=0.07,
	n_estimators=50, 
	subsample=1.0,
	criterion='friedman_mse',
	min_samples_split=2,
	min_samples_leaf=2,
	min_weight_fraction_leaf=0.0,
	max_depth=10,
	min_impurity_decrease=0.0,
	min_impurity_split=None,
	init=None,
	random_state=0,
	max_features=None,
	verbose=0,
	max_leaf_nodes=None,
	warm_start=False,
	presort='auto',
	validation_fraction=0.1,
	n_iter_no_change=None,
	tol=0.0001
)\
.fit(X_train, y_train)
test_model(gbc_model, 'GradientBoostingClassifier')
joblib.dump(gbc_model, 'trained_models/gbc_model.pkl')

