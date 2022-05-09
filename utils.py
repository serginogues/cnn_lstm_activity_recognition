from config import *
import numpy as np
from os import listdir
from os.path import join, isdir
from tqdm import tqdm
import cv2


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

    class_idx = 0
    # iterate through class folders, one folder per class
    for f in tqdm(sorted(listdir(dataset_path)), desc='Creating dataset'):
        class_path = join(dataset_path, f)
        if isdir(class_path):
            for vid in listdir(class_path):
                if vid.endswith(".mp4"):
                    video_path = join(class_path, vid)
                    clips.extend(extract_frames(video_path))
                    labels.append(class_idx)
            class_idx +=1

    return np.asarray(clips), np.asarray(labels)


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
    list_clips = []
    clip = np.zeros(shape=(BATCH_INPUT_SHAPE, 256, 256, 1))

    # create video capture
    vidcap = cv2.VideoCapture(path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    # num_clips = int((total_frames / stride) / BATCH_INPUT_SHAPE)

    cnt = 0
    # run through all video frames
    for idx in range(total_frames):

        # read next frame
        _, frame = vidcap.read()

        # do something with temporal stride
        if idx % stride == 0:
            # reshape and normalize frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256)) / 256.0
            # gray = np.reshape(gray, (256, 256, 1))

            # add frame to clip
            clip[cnt, :, :, 0] = gray
            cnt += 1
            if cnt == BATCH_INPUT_SHAPE:
                list_clips.append(np.copy(clip))
                cnt = 0

    vidcap.release()
    return list_clips
