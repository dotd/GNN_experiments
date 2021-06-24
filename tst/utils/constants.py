# Implementation from https://github.com/gordicaleksa/pytorch-GAT

import os
import enum

# Supported datasets - currently only Cora
class DatasetType(enum.Enum):
    CORA = 0,
    PPI = 1


# Networkx is not precisely made with drawing as it's main feature but I experimented with it a bit
class GraphVisualizationTool(enum.Enum):
    NETWORKX = 0,
    IGRAPH = 1


# Support for 3 different GAT implementations - we'll profile each one of these in playground.py
class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2


# 3 different model training/eval phases used in train.py
class LoopPhase(enum.Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2


class VisualizationType(enum.Enum):
    ATTENTION = 0,
    EMBEDDINGS = 1,
    ENTROPY = 2,


# Global vars used for early stopping. After some number of epochs (as defined by the patience_period var) without any
# improvement on the validation dataset (measured via accuracy/micro-F1 metric), we'll break out from the training loop.
BEST_VAL_PERF = 0
BEST_VAL_LOSS = 0
PATIENCE_CNT = 0


BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'checkpoints')
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

# Make sure these exist as the rest of the code assumes it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

#
# PPI specific information
#

PPI_PATH = os.path.join(DATA_DIR_PATH, 'ppi')
PPI_URL = 'https://data.dgl.ai/dataset/ppi.zip'  # preprocessed PPI data from Deep Graph Library

PPI_NUM_INPUT_FEATURES = 50
PPI_NUM_CLASSES = 121




