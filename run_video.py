import argparse
import numpy as np
from keras.models import load_model
import cv2
import time
from utils import preprocess_frame, read_dataset_classes
from network import *


def run_video(path: str):
    """
    Evaluate video
    """
    VIDEO_PATH = path
    classes = read_dataset_classes(DATASET_PATH)
    model = create_cnn_lstm(len(classes))

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

    # read frame by frame until video is completed
    while vid.isOpened():

        ret, frame = vid.read()  # capture frame-by-frame
        if not ret: break
        counter += 1
        print("Frame #", counter)

        if counter % STRIDE == 0:

            new_frame = preprocess_frame(frame)
            clip.append(new_frame)
            if len(clip) == BATCH_INPUT_SHAPE:
                model_output = model.predict(np.expand_dims(clip, axis = 0))[0]
                predicted_label = classes[np.argmax(model_output)]
                print(predicted_label)
                clip.pop(0)

        cv2.putText(frame, str(predicted_label), (50, 100), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        # checking video frame rate
        start_time = time.time()
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        cv2.putText(frame, "fps: " + str(fps), (50, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

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
    print("Time taken: {0} seconds".format(seconds))
    print("Number of frames: {0}".format(counter))
    fps = counter / seconds
    print("Estimated frames per second: {0}".format(fps))

    # When everything done, release the video capture object
    vid.release()


def main(config):
    path = config.path
    run_video(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to video')
    config = parser.parse_args()
    main(config)