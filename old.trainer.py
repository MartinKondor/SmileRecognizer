import os
import gc
import numpy as np
import tqdm
from PIL import Image
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.externals import joblib


check_random_state(0)
DATA_ROW_LIMIT = 0
files = os.listdir("data")
X, y = [], []
data_scaler, class_encoder = None, None


# reading in data from files and resize them to the same size
print("Loading in images ...")
for file in tqdm.tqdm(files[-DATA_ROW_LIMIT:]):

    # resizing image
    img = Image.fromarray(np.loadtxt("data/" + file, delimiter="\t"))\
        .resize((263, 255), Image.ANTIALIAS)

    # converting img to numpy.ndarray 
    img_data = np.array(img)\
        .reshape(67065)  # 263 * 255

    X.append(img_data)


X = np.array(X)
y = np.loadtxt("target.txt")


# preprocessing
print("Preprocessing images ...")
data_scaler = MinMaxScaler()
class_encoder = OneHotEncoder(categories="auto")

X = data_scaler.fit_transform(X)
y = class_encoder.fit_transform(y.reshape(-1, 1))\
    .toarray()


# save processers
joblib.dump(data_scaler, "trained_models/processing/data_scaler.pkl")
joblib.dump(class_encoder, "trained_models/processing/class_encoder.pkl")

print("Separating train and test sets ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)


# free up some memory
del DATA_ROW_LIMIT, files, X, y, data_scaler, class_encoder, tqdm, Image, \
    check_random_state, train_test_split, OneHotEncoder, MinMaxScaler
gc.collect()


print("Building model ...")
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    learning_rate_init=0.0001,
    max_iter=100,
    random_state=0
)

print("Training model ...")
mlp.fit(X_train, y_train)

print()
print("Accuracy on train set:", mlp.score(X_train, y_train))
print("Accuracy on test set:", mlp.score(X_test, y_test))
print()
print("Saving model ...")
joblib.dump(mlp, "trained_models/current_model.pkl")
