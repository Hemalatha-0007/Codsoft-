"""
evaluate.py
-----------
Computes BLEU-1..BLEU-4 scores for the trained model against a test split,
comparing generated captions to all ground-truth reference captions for
each image.

Usage:
    python evaluate.py
"""

import pandas as pd
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import torch
import config
from dataset import eval_transform
from vocabulary import Vocabulary
from models.model import EncoderCNN, DecoderRNNWithAttention
from predict import load_model, generate_caption_beam_search
from PIL import Image
from tqdm import tqdm


def group_references_by_image(captions_file):
    df = pd.read_csv(captions_file)
    refs = defaultdict(list)
    for _, row in df.iterrows():
        refs[row["image"]].append(Vocabulary.tokenize(row["caption"]))
    return refs


def main():
    device = config.DEVICE
    encoder, decoder, vocab = load_model(device)
    refs_by_image = group_references_by_image(config.CAPTIONS_FILE)

    references, hypotheses = [], []
    image_names = list(refs_by_image.keys())

    print(f"Evaluating on {len(image_names)} images...")
    for img_name in tqdm(image_names):
        image_path = f"{config.IMAGE_DIR}/{img_name}"
        image = Image.open(image_path).convert("RGB")
        image_tensor = eval_transform(image)

        seq, _ = generate_caption_beam_search(encoder, decoder, image_tensor, vocab, device)
        words = vocab.denumericalize(seq[1:])

        hypotheses.append(words)
        references.append(refs_by_image[img_name])

    smoothie = SmoothingFunction().method4
    for n in range(1, 5):
        weights = tuple((1.0 / n if i < n else 0.0) for i in range(4))
        score = corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smoothie)
        print(f"BLEU-{n}: {score:.4f}")


if __name__ == "__main__":
    main()
