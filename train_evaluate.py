import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.utils.layer_utils import count_params
from network import *
from dataset_utils import *

np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


def plot_history(train_history, metric_1: str, metric_2: str):
    M1 = train_history.history[metric_1]
    M2 = train_history.history[metric_2]

    epochs = range(len(M1))

    plt.plot(epochs, M1, 'blue', label=metric_1)
    plt.plot(epochs, M2, 'red', label=metric_2)
    plt.legend()
    plt.show()


def train_evaluate(display: bool = False):
    """
    Train A model and evaluate
    """

    if TRAIN_DATASET == eDatasets.UCF:
        clips_train, clips_test, labels_train, labels_test, classes = create_UCF50()
    elif TRAIN_DATASET == eDatasets.HockeyFights:
        clips_train, clips_test, labels_train, labels_test, classes = create_HockeyFights()
    elif TRAIN_DATASET == eDatasets.RealLifeViolence:
        clips_train, clips_test, labels_train, labels_test, classes = create_RealLifeViolence()
    elif TRAIN_DATASET == eDatasets.ViolentFlow:
        clips_train, clips_test, labels_train, labels_test, classes = create_ViolentFlow()
    else:
        clips_train, clips_test, labels_train, labels_test, classes = create_UCF50()

    print("Training clips: " + str(clips_train.shape))
    print("Testing clips: " + str(clips_test.shape))

    for ARCH_TYPE in range(2):
        model = get_LRCN(len(classes)) if ARCH_TYPE == 0 else get_CNN_BiLSTM(len(classes))
        prefix_name = 'LRCN' if ARCH_TYPE == 0 else 'CNN_BiLSTM'

        early_stop = EarlyStopping(monitor='val_accuracy', patience=15, mode='max', restore_best_weights=True)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        train_hist = model.fit(x=clips_train, y=labels_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                               shuffle=True, callbacks=[early_stop], verbose=1,
                               validation_data=(clips_test, labels_test))
        evaluate_hist = model.evaluate(clips_test, labels_test)
        ev_loss, ev_acc = evaluate_hist

        LOGS = [str(TRAIN_DATASET).split(".")[1],
                '_'.join(classes),
                prefix_name,
                str(np.round(train_hist.history['loss'][-1], 4)),
                str(np.round(train_hist.history['accuracy'][-1], 4)),
                str(np.round(ev_loss, 4)),
                str(np.round(ev_acc, 4)),
                str(count_params(model.trainable_weights)),
                str(len(train_hist.history['loss'])),
                str(BATCH_SIZE),
                str(BATCH_INPUT_LENGTH),
                str(IMAGE_SIZE)]

        if float(ev_acc) > 0.6:
            model_name = '_'.join([str(x) for x in LOGS]) + '.h5'
            model.save('backup/' + model_name)
            print(model_name)

        text = ' '.join([str(x) for x in LOGS])
        f = open('results.txt', 'a')
        f.write('\n')
        f.write(text)
        f.close()

        if display:
            correct = 0
            for idx, clip in enumerate(clips_test):
                y_pred = np.argmax(model.predict(np.expand_dims(clip, axis = 0), verbose=0)[0])
                label = np.argmax(labels_test[idx])
                if y_pred == label:
                    correct += 1
            print("Correct predictions: " + str((correct*100)/len(clips_test)))

            epochs = range(len(train_hist.history['loss']))
            plt.plot(epochs, train_hist.history['loss'], label='loss')
            plt.plot(epochs, train_hist.history['val_loss'], label='val_loss')
            plt.plot(epochs, train_hist.history['accuracy'], label='accuracy')
            plt.plot(epochs, train_hist.history['val_accuracy'], label='val_accuracy')
            plt.legend()
            plt.show()


if __name__ == '__main__':
    train_evaluate()
