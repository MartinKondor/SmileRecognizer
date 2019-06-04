import gc
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.datasets import get_data_home
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.decomposition import PCA
from sklearn.externals import joblib


DATA_DIR = os.path.join(get_data_home(), 'lfw_home', 'lfw_funneled')
IMG_SHAPE = (90, 80)
NUMBER_OF_IMG_TO_LOAD = 600

if not os.path.isdir(DATA_DIR):
    print('Downloading the dataset ...')


# fill lists
NON_SMILE_IMAGES, SMILE_IMAGES = [], []

print('Load non smile list ...')
with open(os.path.join('data', 'NON-SMILE_list.txt'), 'r') as nonsmile_file:
    NON_SMILE_IMAGES = [line for line in nonsmile_file.read().splitlines() if line]

print('Load smile list ...')
with open(os.path.join('data', 'SMILE_list.txt'), 'r') as smile_file:
    SMILE_IMAGES = [line for line in smile_file.read().splitlines() if line]


print('Load & prepare images ...')
print('Number of NON_SMILE_IMAGES', len(NON_SMILE_IMAGES))
print('Number of SMILE_IMAGES', len(SMILE_IMAGES))


def loadimg(flist, i):
    global IMG_SHAPE

    img_filename = flist[i]
    img_path = '_'.join(flist[i].split('_')[:2])
    filename = os.path.join(DATA_DIR, img_path, img_filename)

    return np.array(Image.open(filename) \
                    .convert('L') \
                    .resize(IMG_SHAPE)) \
                    .reshape(IMG_SHAPE[0]*IMG_SHAPE[1])


X = []
y = []


# load non smiling images
for i in range(len(NON_SMILE_IMAGES)):
    if i > (NUMBER_OF_IMG_TO_LOAD // 2):
        break
    try:
        X.append(loadimg(NON_SMILE_IMAGES, i))
        y.append(0)
    except FileNotFoundError:
        continue


# load smiling images
for i in range(len(SMILE_IMAGES)):
    if i > (NUMBER_OF_IMG_TO_LOAD // 2):
        break
    try:
        X.append(loadimg(SMILE_IMAGES, i))
        y.append(1)
    except FileNotFoundError:
        continue


# convert them to numpy.ndarray
X = np.array(X)
y = np.array(y)

"""
# print some data
plt.subplot(121)
plt.imshow(X[0], cmap='gray')
plt.title('Non smiling face')

plt.subplot(122)
plt.imshow(X[-1], cmap='gray')
plt.title('Smiling face')
plt.show()
"""

# free up memory
del NON_SMILE_IMAGES, SMILE_IMAGES, DATA_DIR, NUMBER_OF_IMG_TO_LOAD
gc.collect()

# preprocessing
print('Preprocessing ...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0)


encoder = OneHotEncoder(categories='auto')
scaler = Normalizer()

X_train_scaled = scaler.fit_transform(np.array(X_train, dtype=np.float64))
X_test_scaled = scaler.transform(np.array(X_test, dtype=np.float64))
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.reshape(-1, 1)).toarray()

del X_train, X_test, y_train, y_test
gc.collect()


pca = PCA(n_components=20, whiten=True, random_state=1)
pca.fit(X_train_scaled)

X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


del X_train_scaled, X_test_scaled
gc.collect()


"""
# print some components
fig, axes = plt.subplots(3, 5, figsize=(8, 6), subplot_kw={'xticks':(),'yticks':()})
for component, ax in zip(pca.components_, axes.ravel()):
    ax.imshow(component.reshape(IMG_SHAPE), cmap='viridis')
plt.show()
"""

print(len(X_train_pca), 'train images')
print(len(X_test_pca), 'test images')

print('Training model ...')
from sklearn.neural_network import MLPClassifier


model = MLPClassifier(
    hidden_layer_sizes=[50],
    max_iter=1000,
    learning_rate_init=0.9,
    alpha=0.2,
    random_state=6
)
model.fit(X_train_pca, y_train_encoded)

print('Accuracy on train set:', model.score(X_train_pca, y_train_encoded))
print('Accuracy on test set:', model.score(X_test_pca, y_test_encoded))

print('Saving models ...')
joblib.dump(encoder, 'trained/class_encoder.pkl')
joblib.dump(scaler, 'trained/data_normalizer.pkl')
joblib.dump(pca, 'trained/pca.pkl')
joblib.dump(model, 'trained/mlp_model.pkl')
