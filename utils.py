from config import *
import numpy as np
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import cv2
from tensorflow.keras.utils import to_categorical


def create_dataset(dataset_path: str):
    """
    Parameters
    ----------
    dataset_path
        path to dataset each sub-folder being one class

    Returns
    -------
    list
        A list of clips , 10 frames each (10 x 256 x 256 x 1)
    """
    labels = []
    clips = []
    classes = []

    class_idx = 0
    # iterate through class folders, one folder per class
    for f in sorted(listdir(dataset_path)):
        class_path = join(dataset_path, f)
        if isdir(class_path):
            for vid in tqdm(listdir(class_path), desc="Extracting frames from " + f):
                if vid.endswith(".mp4"):
                    video_path = join(class_path, vid)
                    clips.extend(extract_frames(video_path))
                    labels.append(class_idx)
            class_idx +=1
            classes.append(f)

    return np.asarray(clips), to_categorical(np.asarray(labels)), classes


def extract_frames(path: str, stride: int = 2) -> list:
    """
    Parameters
    ----------
    path
        path to video
    stride
        temporal stride
    Returns
    -------
    list
        list of clips, (BATCH_INPUT_SHAPE x 256 x 256 x 1)
    """

    channels = 1 if USE_GRAY else 3
    clip = np.zeros(shape=(BATCH_INPUT_SHAPE, IMAGE_SIZE, IMAGE_SIZE, channels))

    # create video capture
    vidcap = cv2.VideoCapture(path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # num_clips = int((total_frames / stride) / BATCH_INPUT_SHAPE)

    list_clips = []
    cnt = 0
    # run through all video frames
    for idx in range(total_frames):

        # read next frame
        _, frame = vidcap.read()

        # do something with temporal stride
        if idx % stride == 0:

            color_space = cv2.COLOR_BGR2GRAY if USE_GRAY else cv2.COLOR_BGR2RGB

            # reshape and normalize frame
            gray = cv2.cvtColor(frame, color_space)
            gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE)) / 256.0

            # add frame to clip
            if USE_GRAY:
                clip[cnt, :, :, 0] = gray
            else:
                clip[cnt, :, :, :] = gray
            cnt += 1
            if cnt == BATCH_INPUT_SHAPE:
                list_clips.append(np.copy(clip))
                cnt = 0

    vidcap.release()
    return list_clips
