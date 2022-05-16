# CUSTOM
ARCH_TYPE = 0  # 0 for cnn + lstm, 1 for convLSTM
TEMPORAL_STRIDE = 2
IMAGE_SIZE = 100  # 256
DATA_AUGMENTATION = False
BATCH_INPUT_SHAPE = 12

# TRAIN hyperparams
BATCH_SIZE = 20  # number of training samples per learning iteration
EPOCHS = 15  # number of times the full dataset is seen during training
USE_GRAY = False
seed_constant = 27

# PATHS
TRAIN_DATASET = 'data/HockeyFights/train'
TEST_DATASET = 'data/test_videos'
TEST_MODEL_PATH = 'backup/hockeyFights_cnnlstm_2022_05_12__14_54_Loss0.27_Acc0.92_Stride2_Size100_DataAUGTrue_BatchS12.h5'
VIDEO_EXTENSION = '.mp4'
SAVE = False
