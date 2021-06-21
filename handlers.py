from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from scipy.optimize import curve_fit
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential


def data_prep(df_name, VALUE):
    df_name_cut = df_name[
        #         (df_name.qPE > 95             )&
        #         (df_name.qPE < 200           )&
        (df_name.subeventN <= 1) &
        (df_name.eventTime > 2250) &
        (df_name.eventTime < 2700) &
        (df_name.numEarlyPulses <= 3) &
        (df_name.deltat > 20000) &
        (df_name.neckvetoN == 0) &
        (df_name.fmaxpe < 0.4) &
        #         (df_name.mblikelihoodR > 200  )&
        (df_name.mblikelihoodR < VALUE) &
        (df_name.mblikelihoodZ < 550.0) &
        (df_name.timefit2Z > -1000)
        ]
    return df_name_cut


def FWHM_calculate(X, Y):
    half_max = max(Y) / 2.
    # find when function crosses line half_max (when sign of diff flips)
    # take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    # plot(X[0:len(d)],d) #if you are interested
    # find the left and right most indexes
    left_idx = np.where(d > 0)[0][0]
    right_idx = np.where(d < 0)[-1][0]

    return X[right_idx] - X[left_idx]  # return the difference (full width)


def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])

    return popt


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def errors_plotting(error_SC, error_NN, error_CONVNN, error_MBL, error_TF, ax):
    fig = plt.figure(figsize=(16, 6))
    #     plt.subplot(131)

    n, bins, patches = plt.hist(error_SC, 50, align='mid', color='dodgerblue', edgecolor='k', label='')
    bins = (bins[1:] + bins[:-1]) / 2
    H, A, x01, sigma = gauss_fit(bins, n)
    FWHM1 = 2.35482 * sigma
    plt.plot(bins, gauss(bins, H, A, x01, sigma), '--r', linewidth=2, label='fit')
    plt.xlabel(r'${0}_{1} - {0}_{2}$, мм, FCNN, FWHM = %.2f'.format(ax, '{true}', '{pred}') % round(FWHM1,
                                                                                                    2) + r'$\mu$ = %.2f'.format(
        x01), fontsize=15)
    plt.ylabel('')
    # plt.xlim(-200, 200)
    plt.yticks([])

    #     plt.subplot(132)
    VAR = error_MBL_x
    n, bins, patches = plt.hist(error_NN, 50, align='mid', color='dodgerblue', edgecolor='k', label='')
    bins = (bins[1:] + bins[:-1]) / 2
    H, A, x02, sigma = gauss_fit(bins, n)
    FWHM2 = 2.35482 * sigma
    plt.plot(bins, gauss(bins, H, A, x02, sigma), '--r', linewidth=2, label='fit')
    plt.xlabel(r'${0}_{1} - {0}_{2}$, мм, MBL, FWHM = %.2f'.format(ax, '{true}', '{pred}') % round(FWHM2,
                                                                                                   2) + r'$\mu$ = %.2f'.format(
        x02), fontsize=15)
    plt.ylabel('')
    plt.yticks([])
    # plt.xlim(-200, 200)

    #     plt.subplot(133)
    VAR = error_TF_x
    n, bins, patches = plt.hist(error_CONVNN, 50, align='mid', color='dodgerblue', edgecolor='k', label='')
    bins = (bins[1:] + bins[:-1]) / 2
    H, A, x03, sigma = gauss_fit(bins, n)
    FWHM3 = 2.35482 * sigma
    plt.plot(bins, gauss(bins, H, A, x03, sigma), '--r', linewidth=2, label='fit')
    plt.xlabel(r'${0}_{1} - {0}_{2}$, мм, TF2, FWHM = %.2f'.format(ax, '{true}', '{pred}') % round(FWHM3,
                                                                                                   2) + r'$\mu$ = %.2f'.format(
        x03), fontsize=15)
    plt.ylabel('')
    plt.yticks([])
    # plt.xlim(-200, 200)

    VAR = error_TF_x
    n, bins, patches = plt.hist(error_MBL, 50, align='mid', color='dodgerblue', edgecolor='k', label='')
    bins = (bins[1:] + bins[:-1]) / 2
    H, A, x04, sigma = gauss_fit(bins, n)
    FWHM4 = 2.35482 * sigma
    plt.plot(bins, gauss(bins, H, A, x03, sigma), '--r', linewidth=2, label='fit')
    plt.xlabel(r'${0}_{1} - {0}_{2}$, мм, TF2, FWHM = %.2f'.format(ax, '{true}', '{pred}') % round(FWHM3,
                                                                                                   2) + r'$\mu$ = %.2f'.format(
        x03), fontsize=15)
    plt.ylabel('')
    plt.yticks([])
    # plt.xlim(-200, 200)

    VAR = error_TF_x
    n, bins, patches = plt.hist(error_TF, 50, align='mid', color='dodgerblue', edgecolor='k', label='')
    bins = (bins[1:] + bins[:-1]) / 2
    H, A, x05, sigma = gauss_fit(bins, n)
    FWHM5 = 2.35482 * sigma
    plt.plot(bins, gauss(bins, H, A, x03, sigma), '--r', linewidth=2, label='fit')
    plt.xlabel(r'${0}_{1} - {0}_{2}$, мм, TF2, FWHM = %.2f'.format(ax, '{true}', '{pred}') % round(FWHM3,
                                                                                                   2) + r'$\mu$ = %.2f'.format(
        x03), fontsize=15)
    plt.ylabel('')
    plt.yticks([])
    # plt.xlim(-200, 200)
    #     plt.show()
    return [FWHM1, FWHM2, FWHM3, FWHM4, FWHM5, x01, x02, x03, x04, x05]


def qpe_cut(df_name, thresh):
    if thresh == -1:
        return df_name
    else:
        df_name_cut = df_name[(df_name.qPE < thresh + 100) &
                              (df_name.qPE > thresh)]
        return df_name_cut


def radii_cut(df_name, thresh21, thresh22):
    if thresh21 == thresh22:
        return df_name
    else:
        df_name_cut = df_name[(df_name.X * df_name.X + df_name.Y * df_name.Y + df_name.Z * df_name.Z < thresh22 ** 2) &
                              (df_name.X * df_name.X + df_name.Y * df_name.Y + df_name.Z * df_name.Z > thresh21 ** 2)]
        return df_name_cut


def radii_cut_2(df_name, thresh):
    if thresh == -1:
        return df_name
    else:
        df_name_cut = df_name[
            (df_name.X * df_name.X + df_name.Y * df_name.Y + df_name.Z * df_name.Z < (thresh + 25) ** 2) &
            (df_name.X * df_name.X + df_name.Y * df_name.Y + df_name.Z * df_name.Z > (thresh - 25) ** 2)]
        return df_name_cut


def pred_plotting(y_test, predict, mbl_test, tf2_test, ax, value, qq):
    fig = plt.figure(figsize=(12, 6))
    bins = np.linspace(min(predict), max(predict), 50)
    plt.hist(y_test, bins=bins, label='true sample')
    plt.hist(predict, histtype='step', linewidth=3, bins=bins, label='FCNN')
    plt.hist(mbl_test, histtype='step', linewidth=3, bins=bins, label='mbl')
    plt.hist(tf2_test, histtype='step', linewidth=3, bins=bins, label='TF2')
    plt.xlabel(r'${0}_{1}$, ${0}_{2}$ мм, R < %.f'.format(ax, '{pred}', '{true}') % value, fontsize=22)
    lg = plt.legend(fontsize=18)
    if qq != 0:
        lg.set_title('{0} < qPE < {1}'.format(qq, qq + 100), prop={'size': 'xx-large'})


def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = math.sqrt(XsqPlusYsq + z ** 2) / 850  # r
    elev = math.atan2(math.sqrt(XsqPlusYsq), z)  # theta
    az = math.atan2(y, x)  # phi
    return r, elev, az


def sph2car(phi, costheta, r):
    r = r * 850
    x = r * np.sqrt(1 - costheta ** 2) * np.cos(phi)
    y = r * np.sqrt(1 - costheta ** 2) * np.sin(phi)
    z = r * costheta
    return x, y, z


def dense_model():
    model = Sequential()

    model.add(Dense(units=1000,
                    activation='relu',
                    name='Input_765'))

    model.add(Dense(units=765,
                    activation='relu',
                    name='relu_765nodes'))

    model.add(Dense(units=510,
                    activation='relu',
                    name='relu_510nodes'))

    model.add(Dense(units=255,
                    activation='relu',
                    name='relu_255nodes'))

    model.add(Dense(units=3,
                    name='3nodes_output'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])


def convolutional_model_2D():
    model = Sequential()

    model.add(Input(shape=(128, 128, 3),
                    name='128x128x3'))

    model.add(Conv2D(filters=16,
                     kernel_size=(5, 5),
                     activation='relu',
                     padding="same",
                     input_shape=(128, 128, 3),
                     name='16f_5x5k_relu_same'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding="same",
                           name='2x2p_2x2s_1'))

    model.add(Conv2D(filters=64,
                     kernel_size=(2, 2),
                     activation='relu',
                     padding="same",
                     name='64f_2x2k_relu_same'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding="same",
                           name='2x2p_2x2s_2'))

    model.add(Conv2D(filters=128,
                     kernel_size=(2, 2),
                     activation='relu',
                     padding="same",
                     name='128f_2x2k_relu_same'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding="same",
                           name='2x2p_2x2s_3'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(units=4096,
                    activation='relu',
                    name='relu_4096nodes'))

    model.add(Dense(units=512,
                    activation='relu',
                    name='relu_512nodes'))

    model.add(Dense(units=128,
                    activation='relu',
                    name='relu_128nodes'))

    model.add(Dense(units=32,
                    activation='relu',
                    name='relu_32nodes'))

    model.add(Dense(units=3,
                    name='output'))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    return model


def convolutional_model():
    model = Sequential()

    model.add(Input(shape=(255, 1),
                    name='Input_255'))

    model.add(Conv1D(filters=16,
                     kernel_size=4,
                     strides=2,
                     activation='relu',
                     padding="same",
                     name='relu_16f_k4_s2'))

    model.add(Conv1D(filters=32,
                     kernel_size=4,
                     strides=2,
                     activation='relu',
                     padding="same",
                     name='relu_32f_k4_s2'))

    model.add(Conv1D(filters=64,
                     kernel_size=4,
                     strides=2,
                     activation='relu',
                     padding="same",
                     name='relu_64f_k4_s2'))

    model.add(Conv1D(filters=128,
                     kernel_size=2,
                     strides=1,
                     activation='relu',
                     padding="same",
                     name='relu_128f_k2_s1'))

    model.add(Conv1D(filters=256,
                     kernel_size=2,
                     strides=1,
                     activation='relu',
                     padding="same",
                     name='relu_256f_k2_s1'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(units=255,
                    activation='relu',
                    name='relu_255nodes'))

    model.add(Dense(units=3,
                    name='3nodes_output'))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    return model


def shortcut_block(X_input, N):
    input_shape = (255,)
    X_input = layers.Input(shape=input_shape,
                           name='Input_765')
    X = layers.BatchNormalization(name='block_%i' % N)(X_input)

    X = layers.Dense(units=1000,
                     activation='relu',
                     name='relu_block_%i_1000nodes' % N)(X)
    X = layers.Dense(units=255,
                     activation='relu',
                     name='relu_block_%i_255nodes' % N)(X)
    X = layers.Dense(units=3,
                     activation='relu',
                     name='relu_block_%i_3nodes' % N)(X)
    X = layers.Dense(units=255,
                     activation='relu',
                     name='relu_block_%i_765nodes' % N)(X)
    return X


def SC_model():
    input_shape = (255,)
    X_input = layers.Input(shape=input_shape,
                           name='Input_765')
    N = 1
    X1 = shortcut_block(X_input, N)
    X = layers.add([X1, X_input], name='add_from_Input')
    N = 2
    X2 = shortcut_block(X, N)
    X = layers.add([X1, X2], name='add_from_block1')
    N = 3
    X3 = shortcut_block(X, N)
    X = layers.add([X2, X3],
                   name='add_from_block2')
    X = layers.BatchNormalization(name='output_block')(X)
    X = layers.Dense(units=255,
                     activation='linear',
                     name='linear_255nodes')(X)
    X = layers.Dense(units=3,
                     activation='linear',
                     name='linear_3nodes_output')(X)
    SCmodel = models.Model(X_input, X)
    SCmodel.compile(loss='MSE',
                    optimizer='adam',
                    metrics=['accuracy'])
    return SCmodel
