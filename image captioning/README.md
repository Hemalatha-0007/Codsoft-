# Image Captioning: CNN Encoder + Attention-LSTM / Transformer Decoder

An image captioning system that combines a pre-trained CNN for visual feature
extraction with a sequence model (LSTM with attention, or a Transformer
decoder) to generate natural-language captions.

## Architecture

```
   Image (224x224x3)
         |
   Pretrained CNN (ResNet50 or VGG16, ImageNet weights, frozen by default)
         |
   Spatial feature grid: 14x14x2048  ->  reshaped to 196 x 2048
         |
   ┌─────┴─────────────────────────────┐
   │                                    │
   LSTM decoder w/ Bahdanau attention   Transformer decoder
   (models/model.py)  <-- default       (models/transformer_model.py) <-- bonus
   │                                    │
   └─────┬─────────────────────────────┘
         |
   Generated caption, one word at a time
   (beam search or greedy at inference)
```

**Why this design:**
- **CNN encoder**: ResNet50 / VGG16 pretrained on ImageNet already encode
  rich visual concepts (objects, textures, scenes). We keep the last
  convolutional feature *map* (not the final pooled vector) so the decoder
  can look at different spatial regions of the image as it generates each
  word — this is what makes the attention mechanism meaningful.
- **LSTM + attention decoder** (default, `models/model.py`): implements
  ["Show, Attend and Tell"](https://arxiv.org/abs/1502.03044) — at every
  decoding step, additive attention computes a weighted combination of the
  196 spatial feature vectors, conditioned on the decoder's current hidden
  state, letting the model "look" at relevant image regions per word.
- **Transformer decoder** (bonus, `models/transformer_model.py`): a drop-in
  alternative that treats the CNN feature grid as cross-attention memory,
  using `nn.TransformerDecoder` with causal self-attention over the caption
  so far — no recurrence, trains in parallel over time steps.

## Project structure

```
image_captioning/
├── config.py                 # all hyperparameters & paths in one place
├── vocabulary.py              # tokenizer + word<->index vocabulary
├── dataset.py                 # PyTorch Dataset, transforms, collate fn
├── models/
│   ├── model.py                # EncoderCNN + Attention + LSTM decoder
│   └── transformer_model.py    # bonus: Transformer decoder alternative
├── train.py                   # training loop (LSTM+attention variant)
├── predict.py                 # beam-search inference + attention visualization
├── evaluate.py                # BLEU-1..4 scoring on a captions file
├── prepare_flickr8k.py        # converts raw Flickr8k tokens file to CSV
├── utils.py                   # checkpointing, gradient clipping, AverageMeter
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"   # needed by evaluate.py
```

### Dataset

Designed for **Flickr8k** or **Flickr30k** (easiest to get running end to
end on a single GPU), but any dataset in this flat CSV format works,
including a reduced MS-COCO subset:

```
data/flickr8k/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   └── ...
└── captions.txt      # columns: image,caption (5 caption rows per image)
```

Download Flickr8k (e.g. the Kaggle "Flickr8k" dataset, which already ships
`captions.txt` in this format) and point `config.DATASET_DIR` at it. If you
instead have the original `Flickr8k.token.txt` format, convert it first:

```bash
python prepare_flickr8k.py --tokens_file Flickr8k.token.txt --out data/flickr8k/captions.txt
```

## Train

```bash
python train.py
```

This will:
1. Build a vocabulary from the captions (words appearing < `FREQ_THRESHOLD`
   times become `<UNK>`), saved to `checkpoints/vocab.pkl`.
2. Split off `VALID_SPLIT` of the data for validation.
3. Train the CNN-frozen encoder + attention-LSTM decoder with teacher
   forcing, cross-entropy loss, and a doubly-stochastic attention
   regularizer (encourages the model to eventually attend to every region
   of the image, per the original paper).
4. Save the best checkpoint (by validation loss) to
   `checkpoints/best_model.pth`.

Key knobs in `config.py`:
- `ENCODER_TYPE`: `"resnet50"` (2048-dim features) or `"vgg16"` (512-dim).
- `ENCODER_FINE_TUNE`: set `True` to unfreeze the CNN's last block once the
  decoder has stabilized — usually boosts quality but needs a lower LR and
  more GPU memory.
- `BEAM_SIZE`, `MAX_CAPTION_LEN`, `EMBED_SIZE`, `DECODER_HIDDEN_SIZE`, etc.

## Generate a caption

```bash
python predict.py --image path/to/your_image.jpg
python predict.py --image path/to/your_image.jpg --visualize   # saves attention heatmaps
```

`--visualize` overlays, for each generated word, the region of the image
the model attended to — a nice sanity check that the model is "looking"
at the right things (e.g. attending to the dog when it generates "dog").

## Evaluate

```bash
python evaluate.py
```

Runs beam-search generation over every image in `CAPTIONS_FILE` and reports
corpus-level BLEU-1 through BLEU-4 against the human reference captions.

## Using the Transformer decoder instead

`models/transformer_model.py` implements a parallel-training Transformer
decoder as a drop-in alternative to the LSTM. It reuses the same
`EncoderCNN`, treating the 196 spatial feature vectors as cross-attention
memory. See the usage notes at the bottom of that file for the small
changes needed in the training loop (no per-timestep Python loop, a causal
mask instead, and `ignore_index` on the loss for padding). It's included as
a reference implementation to swap in — `train.py` uses the LSTM+attention
version by default since it trains fine on a single GPU and its attention
maps are easy to visualize for a class project or demo.

## Notes & tips

- With a single GPU, Flickr8k (~8,000 images, 5 captions each) trains a
  decent model in 15-20 epochs, a few hours on a T4/RTX-class GPU.
- If loss plateaus early, try `ENCODER_FINE_TUNE = True` after ~10 epochs
  of decoder-only training (a common two-phase recipe).
- BLEU scores in the 0.55-0.65 (BLEU-1) / 0.15-0.20 (BLEU-4) range are
  typical for this architecture on Flickr8k.
- For larger vocabularies/datasets (e.g. MS-COCO), consider raising
  `FREQ_THRESHOLD`, `BATCH_SIZE`, and switching to the Transformer decoder
  for faster training throughput.
