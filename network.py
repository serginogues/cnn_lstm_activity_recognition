from keras.layers import *
from keras.models import Sequential
from config import *


def create_conv_lstm(num_classes: int):
    """
    Parameters
    ----------
    num_classes
        number of classes

    Returns
    -------
    model
        ConvLSTM model
    """

    # define model
    model = Sequential()

    # 1
    channels = 1 if USE_GRAY else 3
    model.add(ConvLSTM2D(filters=4, kernel_size=3, activation='tanh', data_format='channels_last',
                         recurrent_dropout=0.2, return_sequences=True,
                         input_shape=(BATCH_INPUT_LENGTH, IMAGE_SIZE, IMAGE_SIZE, channels)))
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


def create_cnn_lstm(num_classes: int):
    """
    Parameters
    ----------
    num_classes
        number of classes

    Returns
    -------
    model
        CNN-LSTM model
    """

    # define model
    model = Sequential()

    # CNN feature extraction
    channels = 1 if USE_GRAY else 3
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding="same", activation='relu'),
                              input_shape=(BATCH_INPUT_LENGTH, IMAGE_SIZE, IMAGE_SIZE, channels)))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding="same", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), padding="same", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

    model.add(TimeDistributed(Flatten()))

    # LSTM
    model.add(LSTM(32))

    model.add(Dense(num_classes, activation='softmax'))

    # print model
    model.summary()

    return model
