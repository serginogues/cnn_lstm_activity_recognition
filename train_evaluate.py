import random
from os.path import join
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from config import *
from utils import create_dataset

np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


def get_model(num_classes: int):
    """
    Parameters
    ----------
    num_classes
        number of classes

    Returns
    -------
    model
        keras model
    """

    # define model
    model = Sequential()

    # 1
    channels = 1 if USE_GRAY else 3
    model.add(ConvLSTM2D(filters=4, kernel_size=3, activation='tanh', data_format='channels_last',
                         recurrent_dropout=0.2, return_sequences=True,
                         input_shape=(BATCH_INPUT_SHAPE, IMAGE_SIZE, IMAGE_SIZE, channels)))
    model.add(MaxPool3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    # 2
    model.add(ConvLSTM2D(filters=8, kernel_size=3, activation='tanh', data_format='channels_last',
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPool3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    # 3
    model.add(ConvLSTM2D(filters=14, kernel_size=3, activation='tanh', data_format='channels_last',
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPool3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(TimeDistributed(Dropout(0.2)))

    # 4
    model.add(ConvLSTM2D(filters=16, kernel_size=3, activation='tanh', data_format='channels_last',
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(MaxPool3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))

    # 5
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    # print model
    model.summary()

    return model


def train_evaluate():
    """
    Train a model and evaluate
    """
    clips, labels, classes = create_dataset(DATASET_PATH)
    clips_train, clips_test, labels_train, labels_test = train_test_split(clips, labels, test_size=TEST_SIZE,
                                                                          shuffle=True, random_state=seed_constant)
    model = get_model(len(classes))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    train_hist = model.fit(x=clips_train, y=labels_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                           shuffle=True, validation_split=0.2, callbacks=[early_stop], verbose=1)
    evaluate_hist = model.evaluate(clips_test, labels_test)
    ev_loss, ev_acc = evaluate_hist

    model_name = f'model_{dt.datetime.strftime(dt.datetime.now(), "%Y_%m_%d__%H_%M")}__Loss_{ev_loss}__Accuracy_{ev_acc}.h5'
    model.save('backup/' + model_name)


if __name__ == '__main__':
    train_evaluate()
