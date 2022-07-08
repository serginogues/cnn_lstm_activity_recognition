from enum import Enum

BATCH_INPUT_LENGTH = 10
IMAGE_SIZE = 256  # 256
BATCH_SIZE = 15  # number of training samples per learning iteration
EPOCHS = 100  # number of times the full dataset is seen during training

# keep like this
ONE_CLIPXVIDEO = True
TEMPORAL_STRIDE = 10
DATA_AUGMENTATION = False
USE_GRAY = False
seed_constant = 30
VIDEO_EXTENSION = ['.mp4', '.avi', '.mpg']


class eDatasets(Enum):
    HockeyFights = 1
    RealLifeViolence = 2
    ViolentFlow = 3
    FightDetectionSurv = 4
    UCF = 5
    Custom = 6


TRAIN_DATASET = eDatasets.Custom
