from utils import *
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.utils import to_categorical


def create_HockeyFights():
    """
    Returns
    -------
    arrays
        clips_train, clips_test, labels_train, labels_test, classes
    """

    # path to Hockey Fights Dataset with two sub-folders: fights, nofights
    path = 'C:/Users/azken/Documents/Datasets/Activity Recognition/HockeyFights'

    clips, labels, classes = create_dataset(dataset_path=path, data_aug=False)
    clips_train, clips_test, labels_train, labels_test = train_test_split(clips, labels, test_size=0.1,
                                                                          shuffle=True, random_state=seed_constant)
    return clips_train, clips_test, labels_train, labels_test, classes


def create_UCF50(num_classes: int = 10, train_split: int = 0.9):
    """
    Create custom UCF50 Dataset by randomly picking [num_classes] action categories.
    Then split dataset into train and test without data leakage (respecting video groups)

    Returns
    -------
    arrays
        clips_train, clips_test, labels_train, labels_test, classes
    """
    dataset_path = 'C:/Users/azken/Documents/Datasets/Activity Recognition/UCF50'

    labels_train = []
    labels_test = []
    clips_train = []
    clips_test = []

    classes = []
    class_idx = 0
    selected_classes = [random.randint(1, 50) for x in range(0, num_classes)]

    folder_cnt = 0
    # iterate through class folders
    for f in sorted(listdir(dataset_path)):
        class_path = join(dataset_path, f)

        if isdir(class_path) and folder_cnt in selected_classes:

            class_groups = []
            group_count = 0
            single_group_list = []
            # iterate through videos
            for idx, vid in enumerate(listdir(class_path)):
                group_idx = int(vid.split('_')[2][1:]) - 1

                if group_idx == group_count:
                    single_group_list.append(vid)
                else:
                    class_groups.append(single_group_list.copy())
                    group_count += 1
                    single_group_list.clear()
                    single_group_list.append(vid)
            class_groups.append(single_group_list.copy())

            train_len = int(len(class_groups) * train_split)
            for gr_idx, gr_list in tqdm(enumerate(class_groups), desc="Extracting frames from " + f):
                for vid in gr_list:
                    video_path = join(class_path, vid)
                    clip = extract_frames_single_clip(video_path, BATCH_INPUT_LENGTH)

                    if gr_idx < train_len:
                        # add videos to train
                        clips_train.append(clip)
                        labels_train.append(class_idx)
                    else:
                        # add videos to test
                        clips_test.append(clip)
                        labels_test.append(class_idx)

            class_idx += 1
            classes.append(f)
        folder_cnt += 1

    clips_train, labels_train = suffle_two_lists(clips_train, labels_train)
    clips_train = np.asarray(clips_train)
    labels_train = to_categorical(np.asarray(labels_train))

    clips_test, labels_test = suffle_two_lists(clips_test, labels_test)
    clips_test = np.asarray(clips_test)
    labels_test = to_categorical(np.asarray(labels_test))

    return clips_train, clips_test, labels_train, labels_test, classes


def create_RealLifeViolence():
    path = 'C:/Users/azken/Documents/Datasets/Activity Recognition/Real Life Violence Dataset/'
    clips_train, labels_train, classes = create_dataset(dataset_path=path+'train', data_aug=False)
    clips_test, labels_test, _ = create_dataset(dataset_path=path+'test', data_aug=False)

    return clips_train, clips_test, labels_train, labels_test, classes


def create_ViolentFlow():
    path = 'C:/Users/azken/Documents/Datasets/Activity Recognition/ViolentFlow_custom/'
    clips_train, labels_train, classes = create_dataset(dataset_path=path + 'train', data_aug=False)
    clips_test, labels_test, _ = create_dataset(dataset_path=path + 'test', data_aug=False)

    return clips_train, clips_test, labels_train, labels_test, classes


def createFightDetectionSurv():
    pass

