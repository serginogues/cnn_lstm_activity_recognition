import argparse
import numpy as np
from keras.models import load_model
import cv2
import time
from utils import preprocess_frame, read_dataset_classes
from network import *
from os import listdir
from os.path import join, isdir


def run_video(path: str, classes: list, model):
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
            if len(clip) == BATCH_INPUT_SHAPE:
                model_output = model.predict(np.expand_dims(clip, axis = 0))[0]
                predicted_label = classes[np.argmax(model_output)]
                probb = round(float(np.max(model_output)), 2)
                print(f'Current frame {counter}, Output: {predicted_label} - {probb}')
                clip.pop(0)

        cv2.putText(frame, str(predicted_label) + ' ' + str(probb), (50, 100), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        # checking video frame rate
        """start_time = time.time()
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        cv2.putText(frame, "fps: " + str(fps), (50, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)"""

        # show frame
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
    classes = read_dataset_classes(TRAIN_DATASET)
    model = load_model(TEST_MODEL_PATH)

    for f in sorted(listdir(TEST_DATASET)):
        class_path = join(TEST_DATASET, f)
        if isdir(class_path):
            for vid in listdir(class_path):
                if vid.endswith(VIDEO_EXTENSION):
                    video_path = join(class_path, vid)
                    run_video(video_path, classes, model)


if __name__ == '__main__':
    test_all_videos()