import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils.layer_utils import count_params
from network import *
from dataset_utils import *
from run_video import run_video

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


def run_test(path: str, model_path: str, is_test: bool = True):
    clips_test, labels_test, classes = create_dataset(path, data_aug=False)
    model = load_model(model_path)

    if is_test:
        correct = 0
        for idx, clip in enumerate(clips_test):
            model_output = model.predict(np.expand_dims(clip, axis=0), verbose=0)[0]
            y_pred = np.argmax(model_output)
            label = np.argmax(labels_test[idx])
            print()
            print('Label: ' + classes[label])
            for i in range(len(classes)):
                print(classes[i] + ': ' + str(round(float(model_output[i]), 2)))
            if y_pred == label:
                correct += 1
                print('Correct')
            else:
                print('Incorrect')
        print("Correct predictions: " + str((correct * 100) / len(clips_test)))
    else:
        run_video('data/test_videos/20220507_140000_(16)_MADPARVESVESF016SE.MP4',
              classes, model, True)


def train_eval_single(arch_type: int = 0, estop: bool = True, display: bool = False, isrun_video: bool = False) -> str:
    """
    Train a model and evaluate

    Parameters
    ----------
    arch_type
        0 = LRNC, 1 = LRNC with bidirectional units
    estop
        Early Stopping: min val_loss with 10 epoch patience
    display
        plot training and evaluation loss and accuracy
    isrun_video
        if True, runs test video inference at the end of training

    Returns
    -------
    str
        LOGS
    """
    LR = 0.001
    clips_train, clips_test, labels_train, labels_test, classes = mainscript_dataset()

    model = get_LRCN(len(classes)) if arch_type == 0 else get_CNN_BiLSTM(len(classes))
    prefix_name = 'LRCN' if arch_type == 0 else 'CNN_BiLSTM'

    callb = [EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)] if estop else []

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  metrics=['accuracy'])
    train_hist = model.fit(x=clips_train, y=labels_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                           shuffle=True, callbacks=callb, verbose=1,
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

    if isrun_video:
        test_path = 'data/test_videos/20220507_140000_(16)_MADPARVESVESF016SE.MP4'
        run_video(test_path, classes, model, save=True)

    if display:
        epochs = range(len(train_hist.history['loss']))
        plt.plot(epochs, train_hist.history['loss'], label='loss')
        plt.plot(epochs, train_hist.history['val_loss'], label='val_loss')
        plt.plot(epochs, train_hist.history['accuracy'], label='accuracy')
        plt.plot(epochs, train_hist.history['val_accuracy'], label='val_accuracy')
        plt.legend()
        plt.show()

    return text


def compare_architectures():
    """
    Run experiments to compare architectures and hyperparameter influence on performance
    """
    for ARCH_TYPE in range(2):
        text = train_eval_single(ARCH_TYPE, True)
        f = open('results.txt', 'a')
        f.write('\n')
        f.write(text)
        f.close()


if __name__ == '__main__':
    # train_eval_single(isrun_video=True)

    path = 'C:/Users/azken/Documents/Datasets/Activity Recognition/CustomFights/test'
    run_test(path,
             model_path='backup/Custom_fight_noFight_LRCN_0.067_0.9732_0.49_0.7746_589090_14_15_10_256.h5',
             is_test=False)
