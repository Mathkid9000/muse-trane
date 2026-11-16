import torch, os, json
from pathlib import Path
from dotenv import load_dotenv

# Paths
load_dotenv('env.env')
STORAGE_DIR = Path(os.getenv('STORAGE_DIR'))
DATA_DIR = STORAGE_DIR / 'data'
CHECKPOINT_DIR = STORAGE_DIR / 'checkpoints'

VOCAB_LIST_PATH = STORAGE_DIR / 'vocab.json'
VOCAB_LIST_PLOT_PATH = STORAGE_DIR / 'vocab_plot.json'
DATASET_PATH = STORAGE_DIR / 'data.npz'
TOKEN_CACHE_DIR = STORAGE_DIR / 'tokens'   # parsed-but-not-yet-encoded token text, in shards

# Params
SEQ_LEN = 256
BATCH_SIZE = 16
VAL_SPLIT = 0.05