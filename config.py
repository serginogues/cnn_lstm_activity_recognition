# CUSTOM
ARCH_TYPE = 0  # 0 for cnn + lstm, 1 for convLSTM
ONE_CLIPXVIDEO = True
BATCH_INPUT_LENGTH = 25

TEMPORAL_STRIDE = 2
IMAGE_SIZE = 150  # 256
DATA_AUGMENTATION = False

# TRAIN hyperparams
BATCH_SIZE = 10  # number of training samples per learning iteration
EPOCHS = 50  # number of times the full dataset is seen during training
USE_GRAY = False
seed_constant = 27
TRAIN_DATASET = 'data/HockeyFights/train'
TEST_DATASET = 'data/HockeyFights/test'
VIDEO_EXTENSION = ['.mp4', '.avi', '.mpg']

# TEST
RUN_VIDEO_DATASET = 'data/test_videos'
TEST_MODEL_PATH = 'backup/cnnlstm_2022_05_18__14_37_Loss0.05_Acc0.99_Stride2_Size150_DataAUGFalse_BatchS25.h5'
SAVE = False
