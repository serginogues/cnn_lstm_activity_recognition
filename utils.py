from config import *
import numpy as np
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import cv2
from tensorflow.keras.utils import to_categorical


def preprocess_frame(frame):
    """
    Parameters
    ----------
    frame
        cv2 frame

    Returns
    -------
    frame
        preprocessed frame
    """
    color_space = cv2.COLOR_BGR2GRAY if USE_GRAY else cv2.COLOR_BGR2RGB
    channels = 1 if USE_GRAY else 3
    # reshape and normalize frame
    new_frame = cv2.cvtColor(frame, color_space)
    new_frame = cv2.resize(new_frame, (IMAGE_SIZE, IMAGE_SIZE)) / 256.0
    new_frame = np.reshape(new_frame, (IMAGE_SIZE, IMAGE_SIZE, channels))

    return new_frame


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
                if vid.endswith(".avi"):
                    video_path = join(class_path, vid)
                    video_clips = extract_frames(video_path)
                    clips.extend(video_clips)
                    labels.extend([class_idx for x in range(len(video_clips))])
            class_idx +=1
            classes.append(f)

    return np.asarray(clips), to_categorical(np.asarray(labels)), classes


def read_dataset_classes(dataset_path: str):
    """
    Parameters
    ----------
    dataset_path
        path to dataset where every sub-folder is a different class

    Returns
    -------
    list
        A list classes
    """
    classes = []
    # iterate through class folders, one folder per class
    for f in sorted(listdir(dataset_path)):
        class_path = join(dataset_path, f)
        if isdir(class_path):
            classes.append(f)
    return classes


def extract_frames(path: str) -> list:
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
        success, frame = vidcap.read()
        if not success: break

        # do something with temporal stride
        if idx % STRIDE == 0:
            clip[cnt, :, :, :] = preprocess_frame(frame)
            cnt += 1
            if cnt == BATCH_INPUT_SHAPE:
                list_clips.append(np.copy(clip))
                cnt = 0

    return list_clips
