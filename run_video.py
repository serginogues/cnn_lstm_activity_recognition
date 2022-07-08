import numpy as np
from keras.models import load_model
import cv2
import time
from utils import preprocess_frame, get_classes
from network import *
from os import listdir
from os.path import join, isdir


SAVE = False
TEST_MODEL_PATH = 'backup/Custom_fight_noFight_LRCN_0.067_0.9732_0.49_0.7746_589090_14_15_10_256.h5'


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

    model_output = np.array([0, 0])
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
        cv2.putText(frame, 'Current frame:' + str(counter), (50, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
        if counter % TEMPORAL_STRIDE == 0:

            new_frame = preprocess_frame(frame)
            clip.append(new_frame)
            if len(clip) == BATCH_INPUT_LENGTH:
                model_output = model.predict(np.expand_dims(clip, axis = 0))[0]
                # predicted_label = classes[np.argmax(model_output)]
                probb = round(float(np.max(model_output)), 2)
                clip.pop(0)
                print(f'Current frame {counter}, Output: {model_output}')
        for i in range(len(classes)):
            y_coord = 80 if i == 0 else 110
            cv2.putText(frame, classes[i] + ': ' + str(round(float(model_output[i]), 2)), (50, y_coord), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

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


if __name__ == '__main__':
    path = 'C:/Users/azken/Documents/Datasets/Activity Recognition/fight-detection-surv-dataset/train/noFight/nofi012.mp4'
    classes = get_classes('C:/Users/azken/Documents/Datasets/Activity Recognition/fight-detection-surv-dataset/train')
    print(classes)
    model = load_model(TEST_MODEL_PATH)
    run_video(path=path, classes=classes, model=model, save=False)