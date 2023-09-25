# Solution for task 2 (Image Classifier) of lab assignment for FDA SS23 by [TRAYAN TSONEV]

# imports here
import pandas as pd
import numpy as np
#from tensorflow import keras
from keras import layers
from keras import regularizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy
from keras.applications.efficientnet import EfficientNetB3
from keras.applications.efficientnet import EfficientNetB5

# define additional functions here

def train_predict(X_train, y_train, X_test):
    
    # check that the input has the correct shape
    assert X_train.shape == (len(X_train), 6336)
    assert y_train.shape == (len(y_train), 1)
    assert X_test.shape == (len(X_test), 6336)

    # --------------------------

    # ---Prepocessing---

    # convert to np arrays
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()

    # reshape X_train, X_test to orignal img shape so they can be fed to the model
    X_train = X_train.reshape((-1, 44, 48, 3))
    X_test = X_test.reshape((-1, 44, 48, 3))

    # calculate number of different classes
    classes_num = len(np.unique(y_train))

    # validation split on X_train (80:20)
    split = int(0.8 * X_train.shape[0])
    X_train, X_val = X_train[:split], X_train[split:]
    y_train, y_val = y_train[:split], y_train[split:]
    
    
    # ---Model definition---

    img_shape=(44, 48, 3)   # tupel that matches original img size

    # EfficientNetB2: pretrained powerful CNN model
    # ref: https://arxiv.org/abs/1905.11946
        # if too slow change to EfficientNetB3
    submodelB2 = EfficientNetB2(
        include_top=False,      # don't include the pretained layer since it only takes input with dims (300,300,3) and we want to use (44,48,3)
        input_shape=img_shape,  # specify the dimensions of the pics we want to classify
        weights="imagenet",     # load pretrained weights (on ImageNet dataset)
        pooling='max')          # pooling layer takes max and creates a scalar value

    submodelB2.trainable=True      # ensure that lerning can be performed on all layers

    model = Sequential()        # create a sequential model that we can add layers (not really needed for the final solution but I wanted to play around with different layers)
    model.add(submodelB2)          # add EfficientNetB3 base model
    
    model.add(layers.Dense(512,activation='relu',input_dim=512))    #
    model.add(layers.Dropout(.35))                                  # Dropout layer to reduce overfitting
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(classes_num, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))     # Dense layer with that mathes the number of diff classes (outputs a probability vector)

    model.compile(optimizer='adam',                                 # use Adam as the optimization algorithm (based on SGD)
            loss=SparseCategoricalCrossentropy(from_logits=True),   # use SparseCategoricalCrossentropy (good for multi-class classification), from_logits=True because output already represents probability of belonging to each class 
            metrics=['accuracy'])                                   # use the ratio of correctly classified imgs as a guiding metrix during training/validation                

    val_loss_callback = EarlyStopping(          # stop training if no significant improvemnt in validation loss was observed for 3 epoch (helps avoid overfitting)
        monitor='val_loss',                 
        mode='auto',
        patience=3)             # patience value chosen thorugh trial and error

    # best_epoch_callback = ModelCheckpoint("parameters.h5", monitor='val_loss', save_best_only=True, mode='min')     
        # save the paramters from the epoch wehre the model perfomce the best on the val data
        # only seems to work on TensorFlow version 2.9 and below


    # ---Training---

    # train model for a maximum of 15 epochs (chosen through trial/error, more epochs could produce better results but I'm afraid of overfitting)
    # validate against validation set
    model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), callbacks=[val_loss_callback])
    
    # model.load_weights('parameters.h5')     
        # load the paramters from the best performing epoch
        # only seems works on TensorFlow version 2.9 and below


    # ---Prediction---

    # predict() returns an 2D array with shape[1] of size = (number of classes) 
    # each entry = predicted likelihood that image belongs to that specific class
    # argmax() extracts the index of the entry with the highest probability (= class prediction)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # --------------------------

    # test that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),) or y_pred.shape == (len(X_test), 1)

    return y_pred


if __name__ == "__main__":
    # load data (please load data like that and let every processing step happen **inside** the train_predict function)
    # (change path if necessary)
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    # please put everything that you want to execute outside the function here!


