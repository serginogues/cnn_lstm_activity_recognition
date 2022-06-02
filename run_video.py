import numpy as np
from keras.models import load_model
import cv2
import time
from utils import preprocess_frame
from network import *
from os import listdir
from os.path import join, isdir


SAVE = False
TEST_MODEL_PATH = 'backup/cnnlstm_2022_05_25__11_20_Loss0.25_Acc0.91_Stride2_Size100_DataAUGFalse_BatchS10.h5'


def run_video(path: str, classes: list, model, save: bool = True):
    """
    Evaluate video
    """
    print(path)
    VIDEO_PATH = path

    # begin video capture
    # if the input is the camera, pass 0 instead of the video path
    try:
        vid = cv2.VideoCapture(VIDEO_PATH)
    except:
        vid = cv2.VideoCapture(VIDEO_PATH)

    out = None
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if save:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 'XVID'
        out = cv2.VideoWriter("output.mp4", codec, fps, (frame_width, frame_height))

    # init display params
    start = time.time()
    counter = 0
    clip = []
    predicted_label = ''
    probb = 0.0

    # read frame by frame until video is completed
    while vid.isOpened():

        ret, frame = vid.read()  # capture frame-by-frame
        if not ret: break
        counter += 1

        if counter % TEMPORAL_STRIDE == 0:

            new_frame = preprocess_frame(frame)
            clip.append(new_frame)
            if len(clip) == BATCH_INPUT_LENGTH:
                model_output = model.predict(np.expand_dims(clip, axis = 0))[0]
                predicted_label = classes[np.argmax(model_output)]
                probb = round(float(np.max(model_output)), 2)
                print(f'Current frame {counter}, Output: {predicted_label} - {probb}')
                clip.pop(0)

        cv2.putText(frame, str(predicted_label) + ' ' + str(probb), (50, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        # checking video frame rate
        """start_time = time.time()
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        cv2.putText(frame, "fps: " + str(fps), (50, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)"""

        # show frame
        if save:
            out.write(frame)
        else:
            cv2.imshow("Output Video", frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Closes all the frames
    cv2.destroyAllWindows()

    # Average fps
    end = time.time()
    seconds = end - start
    fps = counter / seconds
    print("Estimated frames per second: {0}".format(fps))
    print()

    # When everything done, release the video capture object
    vid.release()


def test_all_videos():
    classes = []
    # iterate through class folders, one folder per class
    for f in sorted(listdir(TRAIN_DATASET)):
        class_path = join(TRAIN_DATASET, f)
        if isdir(class_path):
            classes.append(f)

    model = load_model(TEST_MODEL_PATH)

    test_videos_path = 'data/test_videos'
    for f in sorted(listdir(test_videos_path)):
        class_path = join(test_videos_path, f)
        if isdir(class_path):
            for vid in listdir(class_path):
                if vid.endswith(VIDEO_EXTENSION[0]) or vid.endswith(VIDEO_EXTENSION[1]) or vid.endswith(VIDEO_EXTENSION[2]):
                    run_video(join(class_path, vid), classes, model, SAVE)


if __name__ == '__main__':
    test_all_videos()