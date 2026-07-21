"""
config.py
---------
Central place for every hyperparameter and path used across the project.
Edit this file rather than hard-coding values inside the other scripts.
"""

import torch

# ---------------------------------------------------------------------------
# Paths (adjust to match where you unzip your dataset, e.g. Flickr8k/Flickr30k)
# ---------------------------------------------------------------------------
DATASET_DIR = "data/flickr8k"                       # root dataset folder
IMAGE_DIR = f"{DATASET_DIR}/Images"                 # folder with .jpg images
CAPTIONS_FILE = f"{DATASET_DIR}/captions.txt"       # "image_name,caption" csv

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
VOCAB_PATH = f"{CHECKPOINT_DIR}/vocab.pkl"

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
FREQ_THRESHOLD = 5          # word must appear at least this many times to be kept
MAX_CAPTION_LEN = 35        # captions are truncated/padded to this length (incl. <SOS>/<EOS>)

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
ENCODER_TYPE = "resnet50"   # options: "resnet50", "vgg16"
EMBED_SIZE = 256            # word embedding dimension
ATTENTION_DIM = 256         # dimension of the attention layer
DECODER_HIDDEN_SIZE = 512   # LSTM hidden state size
ENCODER_FINE_TUNE = False   # if True, unfreeze last CNN block for fine-tuning
DROPOUT = 0.5

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
ENCODER_LEARNING_RATE = 1e-4   # only used if ENCODER_FINE_TUNE = True
GRAD_CLIP = 5.0
NUM_WORKERS = 2
VALID_SPLIT = 0.1             # fraction of data held out for validation
PRINT_EVERY = 100             # batches between progress prints

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
BEAM_SIZE = 3                  # beam width used at caption-generation time
