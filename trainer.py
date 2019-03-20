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


check_random_state(5)
files = os.listdir('data')
data = []

# reading in data from files and resize them to the same size
for file in files:
    data.append(np.array(Image.fromarray(np.loadtxt('data/' + file, delimiter='\t'))\
			.resize((300, 168), Image.ANTIALIAS).getdata()))

X = np.array(data)
y = load_targets('data')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


from sklearn.neural_network import MLPClassifier
model = MLPClassifier(
	hidden_layer_sizes=(200,),
	activation='relu',
	solver='adam',
	alpha=0.0001,
	batch_size='auto',
	learning_rate='constant',
	learning_rate_init=0.0001,
	power_t=0.5,
	max_iter=150,
	shuffle=True,
	random_state=3,
	tol=0.0001, 
	verbose=False,
	warm_start=False,
	momentum=0.9,
	nesterovs_momentum=True,
	early_stopping=False,
	validation_fraction=0.1,
	beta_1=0.9,
	beta_2=0.999,
	epsilon=1e-08,
	n_iter_no_change=10
)
model.fit(X_train, y_train)
print('Error on train:', mse(y_train, model.predict(X_train)))
print('Error on test:', mse(y_test, model.predict(X_test)))

joblib.dump(model, 'trained_models/temp_MLPNN_model.pkl')

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(
	n_neighbors=5,
	weights='uniform',
	algorithm='auto',
	leaf_size=30,
	p=2,
	metric='minkowski',
	n_jobs=-1
)
model.fit(X_train, y_train)
print('Error on train:', mse(y_train, model.predict(X_train)))
print('Error on test:', mse(y_test, model.predict(X_test)))

joblib.dump(model, 'trained_models/temp_KNN_model.pkl')



