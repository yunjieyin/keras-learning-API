from os import path

OS_TYPE = "linux"
# Dataset base path which contain a .csv file
BASE_PATH = "/algdata01/yunjie/misc/fer"

# .csv dataset path
INPUT_PATH = path.sep.join([BASE_PATH, "fer2013.csv"])

# define the number os classes
NUM_CLASSES = 6

# Dataset HDF5 files
"""
1. create "hdf5" directory manually in the BASE_PATH, 
    or build_dataset.py will raise error
2. create "output" directory manually in the BASE_PATH
"""
if OS_TYPE == "linux":
    TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
    VAL_HDF5   = path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
    TEST_HDF5  = path.sep.join([BASE_PATH, "hdf5/test.hdf5"])

    OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])
else:
    TRAIN_HDF5 = BASE_PATH + "/hdf5/train.hdf5"
    VAL_HDF5   = BASE_PATH + "/hdf5/val.hdf5"
    TEST_HDF5  = BASE_PATH + "/hdf5/test.hdf5"

    OUTPUT_PATH = BASE_PATH + "/output"

BATCH_SIZE = 128


