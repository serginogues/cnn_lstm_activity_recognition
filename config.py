# CUSTOM
ARCH_TYPE = 0  # 0 for cnn + lstm, 1 for convLSTM
TEMPORAL_STRIDE = 3
IMAGE_SIZE = 100  # 256
DATA_AUGMENTATION = False
BATCH_INPUT_SHAPE = 10

# TRAIN hyperparams
BATCH_SIZE = 20  # number of training samples per learning iteration
EPOCHS = 15  # number of times the full dataset is seen during training
USE_GRAY = False
seed_constant = 27

# PATHS
TRAIN_DATASET = 'data/Real Life Violence Dataset/train'
TEST_DATASET = 'data/test_videos'
TEST_MODEL_PATH = 'backup/cnnlstm_2022_05_12__16_02_Loss0.66_Acc0.79_Stride3_Size100_DataAUGFalse_BatchS10.h5'
VIDEO_EXTENSION = '.mp4'
