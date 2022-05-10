import random
import numpy as np
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from utils import create_dataset
from network import *

np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


def plot_history(train_history, metric_1: str, metric_2: str):
    """
    Parameters
    ----------
    train_history
        tensorflow model training history
    metric_1
        metric to be displayed
    metric_2
        metric to be displayed
    """

    M1 = train_history.history[metric_1]
    M2 = train_history.history[metric_2]

    epochs = range(len(M1))

    plt.plot(epochs, M1, 'blue', label=metric_1)
    plt.plot(epochs, M2, 'red', label=metric_2)
    plt.legend()
    plt.show()


def train_evaluate():
    """
    Train a model and evaluate
    """
    clips, labels, classes = create_dataset(DATASET_PATH)
    clips_train, clips_test, labels_train, labels_test = train_test_split(clips, labels, test_size=TEST_SIZE,
                                                                          shuffle=True, random_state=seed_constant)
    print("Training clips: " + str(clips_train.shape))
    print("Testing clips: " + str(clips_test.shape))

    model = create_cnn_lstm(len(classes)) if ARCH_TYPE == 0 else create_conv_lstm(len(classes))

    early_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    train_hist = model.fit(x=clips_train, y=labels_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                           shuffle=True, validation_split=0.2, callbacks=[early_stop], verbose=1)
    evaluate_hist = model.evaluate(clips_test, labels_test)
    ev_loss, ev_acc = evaluate_hist

    prefix_name = 'cnnlstm' if ARCH_TYPE == 0 else 'convlstm'

    model_name = f'{prefix_name}_{dt.datetime.strftime(dt.datetime.now(), "%Y_%m_%d__%H_%M")}__Loss_{str(np.round(ev_loss, 2))}__Accuracy_{str(np.round(ev_acc, 2))}.h5'
    model.save('backup/' + model_name)

    plot_history(train_hist, 'loss', 'val_loss')
    plot_history(evaluate_hist, 'accuracy', 'val_accuracy')


if __name__ == '__main__':
    train_evaluate()
