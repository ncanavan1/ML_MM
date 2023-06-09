#!/usr/bin/env python3
##IMPORT LIBRARIES##
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, layers, initializers, regularizers
import math 
import csv
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import chi2
import sys

##IMPORT DATASET##
#df = pd.read_csv(<DS-PE>) ##MODEL 1

sf_reduced = sys.argv[1]
modeltype = int(sys.argv[2])
imgprefix = sys.argv[3]
##model number defines type of model, 1 is min energy, 2 is energy diff regression, 3 energy diff classifcation


df = pd.read_csv(sf_reduced) ##MODEL 2/3


##LIMIT ENERGY RANGE  BY DROPPING INVALID ENERGIES

if (modeltype == 1 ):
    #energy_range = [-1800,0] ##MODEL 1(A)
    energy_range = [-1000,0] ##MODEL 1(B)

else:
    energy_range = [0.01,0.5] ##MODEL 2(A),3
#    energy_range = [0.01,0.2] ##MODEL 2(B)

invalid = []
for index, row in df.iterrows():
    if row["Energy"] < energy_range[0] or row["Energy"] > energy_range[1] or row["Molecule"][0:2] != "Dy":
        invalid.append(index)      
df=df.drop(df.index[invalid])


energies=df.iloc[:,-1]
null_energies=[]
for i in range(energies.shape[0]):
    if np.isnan(energies.iloc[i]):
        null_energies.append(i)

df=df.drop(df.index[null_energies])
names=df.iloc[:,0]
energies=df.iloc[:-1]
df_x = df.iloc[:,1:-1]

##GENERATE TRAIN/TEST SPLIT SETS NORMALISED

x=df_x.values

if (modeltype == 1 or modeltype == 2):
    y = df['Energy'].values
    ysc = StandardScaler().fit(y.reshape(-1,1))
    y = ysc.fit_transform(y.reshape(-1,1))
elif (modeltype == 3):
    y = []
    yt = df['Energy'].values
    for i in yt:
        if i > 0 and i < 0.1:
            y.append([1,0,0,0])
        elif i >= 0.1 and i < 0.2:
            y.append([0,1,0,0])
        elif i >= 0.2 and i < 0.3:
            y.append([0,0,1,0])
        else:
            y.append([0,0,0,1])

x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4, random_state=0, shuffle=True)

              
##DEFINE TENSORFLOW MODELS

###Model to predict min energy
def getModel1(n_features):
    model = Sequential()
    model.add(Dense(1000, input_dim = n_features, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(500, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(200, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(50, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(20, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(1, activation='linear', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    opt = tf.optimizers.Adam(lr=0.05)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['RootMeanSquaredError'])
    return model


###Regression model to predict U
def getModel2(n_features):
    model = Sequential()
    model.add(Dense(n_features, input_dim = n_features, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(1000, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(500, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(200, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(50, activation='relu', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    model.add(Dense(1, activation='linear', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(0)))
    opt = tf.optimizers.Adam(lr=0.05)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['RootMeanSquaredError'])
    return model

###Classification model to predict U
def getModel3(n_features):
    model = Sequential()
    model.add(Dense(1000, input_dim = n_features, activation='elu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001),bias_initializer=initializers.Constant(1)))
    model.add(Dense(500, activation='elu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001),bias_initializer=initializers.Constant(1)))
    model.add(Dense(100, activation='elu', kernel_initializer='random_uniform',kernel_regularizer=regularizers.l2(0.001), bias_initializer=initializers.Constant(1)))
    model.add(Dense(4, activation='softmax', kernel_initializer='random_uniform', bias_initializer=initializers.Constant(1)))
    opt = tf.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.categorical_crossentropy
    met = tf.metrics.Accuracy
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    return model



##FIT MODELS

tf.random.set_seed(0)
np.random.seed(0)
y_train=np.asarray(y_train).astype('float32')
y_test=np.asarray(y_test).astype('float32')
x_train=np.asarray(x_train).astype('float32')
x_test=np.asarray(x_test).astype('float32')

if (modeltype == 1):
    model = getModel1(len(x_train[0]))
elif (modeltype == 2):
    model = getModel2(len(x_train[0]))
elif (modeltype == 3):
    model = getModel3(len(x_train[0]))


history = model.fit(x_train, y_train,validation_data=(x_test, y_test), batch_size = 16, epochs=100, verbose=2)

#Evaluating the model on the test data

pred_train = np.asarray(model(x_train))
pred_test = np.asarray(model(x_test))

if (modeltype == 1 or modeltype == 2):

    pred_train = ysc.inverse_transform(pred_train)
    pred_test = ysc.inverse_transform(pred_test)

    ##INVERT SCALAR
    y_train = ysc.inverse_transform(y_train)
    y_test = ysc.inverse_transform(y_test)

    tr=np.arange(0,0.45,0.001) #TRUTH LINE, CHANGE AS APPROPRIATE

    ##TRAIN SCATTER
    plt.figure(1)
    plt.plot(tr,tr,color="r")
    plt.scatter(pred_train,y_train)
    plt.xlabel("Prediciton (a.u)")
    plt.ylabel("Ground Truth (a.u)")
    plt.savefig(str(imgprefix) + "_train.png")

    ##TEST SCATTER
    plt.figure(2)
    plt.plot(tr,tr,color="r")
    plt.scatter(pred_test,y_test)
    plt.xlabel("Prediciton (a.u)")
    plt.ylabel("Ground Truth (a.u)")
    plt.savefig(str(imgprefix) + "_test.png")

    ##LOSS CURVE
    #CHANGE 'loss' TO LOSS USED 
    plt.figure(3)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('mean_squared_error (a.u)')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(str(imgprefix) + "_mean_squared_error.png")

    ##LOSS METRIC
    #CHANGE 'root_mean_squared_error' TO METRIC USED 
    plt.figure(4)
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.ylabel('RMSE (a.u)')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(str(imgprefix) + "_rmse.png")


##MODEL 3 SPECIFIC

def round_4(pred):
    #round = [0,0,0,0]
    round = 0
    if pred[0] == np.max(pred):
        #round = [1,0,0,0]
        round = 1
    elif pred[1] == np.max(pred):
        #round = [0,1,0,0]
        round = 2
    elif pred[2] == np.max(pred):
        #round = [0,0,1,0]
        round = 3
    elif pred[3] == np.max(pred):
        #round = [0,0,0,1]
        round = 4
    return round


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

if (modeltype == 3):

    pred_test = np.asarray(model(x_test))
    pred_train = np.asarray(model(x_train))

    pred_test_round = []
    pred_train_round = []
    for p in pred_test:
        pred_test_round.append(round_4(p))

    for p in pred_train:
        pred_train_round.append(round_4(p))

    ylbtr = []
    ylbtt = []
    for p in y_test:
        ylbtt.append(round_4(p))

    for p in y_train:
        ylbtr.append(round_4(p))

    cm_train = confusion_matrix(ylbtr,pred_train_round)
    cm_test = confusion_matrix(ylbtt,pred_test_round)

    labels = ["0-0.1","0.1-0.2","0.2-0.3","<0.3"]


    ##LOSS CURVE
    #CHANGE 'loss' TO LOSS USED 
    plt.figure(3)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('BCE')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(str(imgprefix) + "_BCE.png")

    ##LOSS METRIC
    #CHANGE 'root_mean_squared_error' TO METRIC USED 
    plt.figure(4)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy (a.u)')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(str(imgprefix) + "_accuracy.png")


    plt.figure(5)
    dispTrain = ConfusionMatrixDisplay(confusion_matrix=cm_train,display_labels=labels)
    dispTrain.plot()
    plt.xlabel("Predicted Energy (a.u)")
    plt.ylabel("True Energy (a.u)")
    plt.savefig(str(imgprefix) + "_train_conf.png")

    plt.figure(6)
    dispTrain = ConfusionMatrixDisplay(confusion_matrix=cm_test,display_labels=labels)
    dispTrain.plot()
    plt.xlabel("Predicted Energy (a.u)")
    plt.ylabel("True Energy (a.u)")
    plt.savefig(str(imgprefix) + "_test_conf.png")
