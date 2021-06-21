from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from handlers import *

models = tf.keras.models
layers = tf.keras.layers
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers
losses = tf.keras.losses
optimizers = tf.keras.optimizers
metrics = tf.keras.metrics
activations = tf.keras.activations
preprocessing_image = tf.keras.preprocessing.image
sc = StandardScaler()

plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22

radii = [0]
for i in range(5, 105, 5):
    radii.append((i / 100) ** (1 / 3) * 851)
print(radii)

radii2 = []
for i in range(0, 900, 50):
    radii2.append(i)
print(radii2)

# Подключаем файл, в котором должны содержаться координаты и пути к входным файлам
print("loading data...")
df1 = pd.read_csv("../allfiles_v2.csv", skiprows=0, sep=",", header=0)
df_raw_raw = pd.read_csv("/home/ailyasov/scratch/produce_slim/df1.csv")

phi = []
costheta = []
r = []
bb = []
for i in range(len(df1)):
    a, b, c = cart2sph(df1['mblX'][i], df1['mblY'][i], df1['mblZ'][i])
    r.append(a)
    phi.append(c)
    costheta.append(np.cos(b))
    bb.append(b)

path = df1.iloc[:, 8]
df = pd.DataFrame(dict(phi=phi, costheta=costheta, r=r))

X = path
y = df1[['mblX', 'mblY', 'mblZ']]
X_train_CONV, X_test_CONV, y_train_CONV, y_test_CONV = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_images = []
X_test_images = []
for path in X_train_CONV:
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    X_train_images.append(image)
for path in X_test_CONV:
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    X_test_images.append(image)

X_train_images = np.array(X_train_images)
X_test_images = np.array(X_test_images)
X_train_images = X_train_images / 255
X_test_images = X_test_images / 255
X_train_images = X_train_images.reshape(X_train_images.shape[0], 128, 128, 3).astype('float32')
X_test_images = X_test_images.reshape(X_test_images.shape[0], 128, 128, 3).astype('float32')

value = 850
df_raw_cut = data_prep(df_raw, value)

lentrain = int(df_raw.shape[0] * 0.7)

df_train = df_raw.iloc[:lentrain]

x_train = df_train.iloc[:, :255]
y_train = df_train[['X', 'Y', 'Z']]
print('train sample shape:', x_train.shape)

mbl_x_train = df_raw['mblikelihoodX'].iloc[:lentrain]
mbl_y_train = df_raw['mblikelihoodY'].iloc[:lentrain]
mbl_z_train = df_raw['mblikelihoodZ'].iloc[:lentrain]

tf2_x_train = df_raw['timefit2X'].iloc[:lentrain]
tf2_y_train = df_raw['timefit2Y'].iloc[:lentrain]
tf2_z_train = df_raw['timefit2Z'].iloc[:lentrain]

# Модель 2D свёрточной нейронной сети
model_conv2d = convolutional_model_2D()

history_2DCONVNN = model_conv2d.fit(x=X_train_images,
                                    y=y_train_CONV,
                                    epochs=30,
                                    batch_size=500,
                                    validation_split=0.3)

# Модель 1D свёрточной нейронной сети
model_conv = convolutional_model()

history_CONVNN = model_conv.fit(x=x_train,
                                y=y_train,
                                epochs=30,
                                batch_size=1000,
                                validation_split=0.3)

# Модель полносвязной нейронной сети

model_fcnn = dense_model()

history_FCNN = model_fcnn.fit(x=x_train,
                              y=y_train,
                              epochs=30,
                              batch_size=1000,
                              validation_split=0.3)

# Модель нейронной сети с Shortcut-связями
model_sc = SC_model()

history_SC = model_sc.fit(x=x_train,
                          y=y_train,
                          epochs=30,
                          batch_size=1000,
                          validation_split=0.3)

df_test = df_raw.iloc[lentrain:]
x_test = df_test.iloc[:, :255]
print('test sample shape:', x_test.shape)
y_test = df_test[['X', 'Y', 'Z']]

predictions_conv2d = model_conv2d.predict(X_test_images)
predictions_conv = model_conv.predict(x_test)
predictions_fcnn = model_fcnn.predict(x_test)
predictions_sc = model_sc.predict(x_test)

a, b, c = zip(*predictions_conv2d)
pred_x_conv2d = list(a)
pred_y_conv2d = list(b)
pred_z_conv2d = list(c)
a, b, c = zip(*predictions_conv)
pred_x_conv = list(a)
pred_y_conv = list(b)
pred_z_conv = list(c)
a, b, c = zip(*predictions_fcnn)
pred_x_fcnn = list(a)
pred_y_fcnn = list(b)
pred_z_fcnn = list(c)
a, b, c = zip(*predictions_sc)
pred_x_sc = list(a)
pred_y_sc = list(b)
pred_z_sc = list(c)

y_test_r = np.sqrt(y_test['X'] ** 2 +
                   y_test['Y'] ** 2 +
                   y_test['Z'] ** 2)

mbl_r_test = np.sqrt(df_test['mblikelihoodX'] ** 2 +
                     df_test['mblikelihoodY'] ** 2 +
                     df_test['mblikelihoodZ'] ** 2)

tf2_r_test = np.sqrt(df_test['timefit2X'] ** 2 +
                     df_test['timefit2Y'] ** 2 +
                     df_test['timefit2Z'] ** 2)

error_TF_x = y_test['X'] - df_test['timefit2X']
error_TF_y = y_test['Y'] - df_test['timefit2Y']
error_TF_z = y_test['Z'] - df_test['timefit2Z']
error_TF_r = np.sqrt(error_TF_x ** 2 + error_TF_y ** 2 + error_TF_z ** 2)

error_MBL_x = y_test['X'] - df_test['mblikelihoodX']
error_MBL_y = y_test['Y'] - df_test['mblikelihoodY']
error_MBL_z = y_test['Z'] - df_test['mblikelihoodZ']
error_MBL_r = np.sqrt(error_MBL_x ** 2 + error_MBL_y ** 2 + error_MBL_z ** 2)

error_x_conv2d = y_test_CONV['mblX'] - pred_x_conv2d
error_y_conv2d = y_test_CONV['mblY'] - pred_y_conv2d
error_z_conv2d = y_test_CONV['mblZ'] - pred_z_conv2d

error_x_conv = y_test['X'] - pred_x_conv
error_y_conv = y_test['Y'] - pred_y_conv
error_z_conv = y_test['Z'] - pred_z_conv

error_x_fcnn = y_test['X'] - pred_x_fcnn
error_y_fcnn = y_test['Y'] - pred_y_fcnn
error_z_fcnn = y_test['Z'] - pred_z_fcnn

error_x_sc = y_test['X'] - pred_x_sc
error_y_sc = y_test['Y'] - pred_y_sc
error_z_sc = y_test['Z'] - pred_z_sc

error_xyz_r_0 = np.sqrt(error_xyz_x_0 ** 2 + error_xyz_y_0 ** 2 + error_xyz_z_0 ** 2)
error_xyz_r_1 = np.sqrt(error_xyz_x_1 ** 2 + error_xyz_y_1 ** 2 + error_xyz_z_1 ** 2)
error_xyz_r_2 = np.sqrt(error_xyz_x_2 ** 2 + error_xyz_y_2 ** 2 + error_xyz_z_2 ** 2)
error_xyz_r_3 = np.sqrt(error_xyz_x_3 ** 2 + error_xyz_y_3 ** 2 + error_xyz_z_3 ** 2)

plt.figure(figsize=(12, 6))
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
bins = np.linspace(-900, 900, 50)
plt.hist(y_test['X'], bins=bins, density=True, label='true sample')
plt.hist(predictions_0[:, 0], histtype='step', linewidth=2, density=True, bins=bins, label='2DCONVNN')
plt.hist(predictions_1[:, 0], histtype='step', linewidth=2, density=True, bins=bins, label='SC')
plt.hist(predictions_2[:, 0], histtype='step', linewidth=2, density=True, bins=bins, label='FCNN')
plt.hist(predictions_3[:, 0], histtype='step', linewidth=2, density=True, bins=bins, label='CONVNN')
plt.xlabel(r'$X_{pred}$, и $X_{true}$ мм для всех моделей при R < 850', fontsize=22)
plt.legend(fontsize=18)
plt.savefig('diff_distr_pred.png')
