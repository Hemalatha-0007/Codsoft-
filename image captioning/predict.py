"""
predict.py
----------
Loads a trained checkpoint and generates a caption for a single image using
beam search. Optionally visualizes where the model "looked" (attention maps)
at each generated word.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --visualize
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import config
from dataset import eval_transform
from vocabulary import Vocabulary
from models.model import EncoderCNN, DecoderRNNWithAttention
from utils import load_checkpoint


def load_model(device):
    vocab = Vocabulary.load(config.VOCAB_PATH)
    checkpoint = load_checkpoint(config.BEST_MODEL_PATH, device)

    encoder = EncoderCNN(config.ENCODER_TYPE, fine_tune=False).to(device)
    decoder = DecoderRNNWithAttention(
        embed_size=config.EMBED_SIZE,
        hidden_size=config.DECODER_HIDDEN_SIZE,
        vocab_size=checkpoint["vocab_size"],
        encoder_dim=encoder.feature_dim,
        attention_dim=config.ATTENTION_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state"])
    decoder.load_state_dict(checkpoint["decoder_state"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab


@torch.no_grad()
def generate_caption_beam_search(encoder, decoder, image_tensor, vocab, device, beam_size=None):
    """
    Beam search caption generation. Returns the best sequence of token ids
    and, for each step, the attention weights used (useful for visualization).
    """
    beam_size = beam_size or config.BEAM_SIZE
    k = beam_size
    vocab_size = len(vocab)

    encoder_out = encoder(image_tensor.unsqueeze(0).to(device))     # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)     # replicate for each beam

    sos_idx = vocab.stoi["<SOS>"]
    eos_idx = vocab.stoi["<EOS>"]

    seqs = torch.full((k, 1), sos_idx, dtype=torch.long, device=device)
    top_k_scores = torch.zeros(k, 1, device=device)
    seqs_alpha = torch.ones(k, 1, num_pixels, device=device)

    complete_seqs, complete_seqs_scores, complete_seqs_alpha = [], [], []

    h, c = decoder.init_hidden_state(encoder_out)
    step = 1

    while True:
        embeddings = decoder.embedding(seqs[:, -1])                  # (k, embed_size)
        context, alpha = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        context = gate * context

        lstm_input = torch.cat([embeddings, context], dim=1)
        h, c = decoder.lstm_cell(lstm_input, (h, c))
        scores = F.log_softmax(decoder.fc(h), dim=1)                 # (k, vocab_size)

        scores = top_k_scores.expand_as(scores) + scores
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, dim=0)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)

        prev_word_idxs = top_k_words // vocab_size
        next_word_idxs = top_k_words % vocab_size

        seqs = torch.cat([seqs[prev_word_idxs], next_word_idxs.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)], dim=1)

        incomplete_idxs = [i for i, w in enumerate(next_word_idxs) if w.item() != eos_idx]
        complete_idxs = list(set(range(len(next_word_idxs))) - set(incomplete_idxs))

        if complete_idxs:
            complete_seqs.extend(seqs[complete_idxs].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_idxs].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_idxs].tolist())
        k -= len(complete_idxs)

        if k == 0 or step > config.MAX_CAPTION_LEN:
            break

        seqs = seqs[incomplete_idxs]
        seqs_alpha = seqs_alpha[incomplete_idxs]
        h = h[prev_word_idxs[incomplete_idxs]]
        c = c[prev_word_idxs[incomplete_idxs]]
        encoder_out = encoder_out[prev_word_idxs[incomplete_idxs]]
        top_k_scores = top_k_scores[incomplete_idxs].unsqueeze(1)
        step += 1

    if not complete_seqs:
        # Fallback: no sequence hit <EOS> in time, just take current best.
        complete_seqs = seqs.tolist()
        complete_seqs_scores = top_k_scores.squeeze(1).tolist()
        complete_seqs_alpha = seqs_alpha.tolist()

    best_idx = int(np.argmax(complete_seqs_scores))
    return complete_seqs[best_idx], complete_seqs_alpha[best_idx]


def visualize_attention(image_path, words, alphas):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    num_words = len(words)
    cols = 5
    rows = (num_words + cols - 1) // cols

    plt.figure(figsize=(cols * 3, rows * 3))
    for t in range(num_words):
        plt.subplot(rows, cols, t + 1)
        plt.imshow(image)
        alpha = np.array(alphas[t]).reshape(14, 14)
        alpha_img = np.array(Image.fromarray(alpha).resize((224, 224), Image.BICUBIC))
        plt.imshow(alpha_img, alpha=0.6, cmap="jet")
        plt.title(words[t])
        plt.axis("off")
    plt.tight_layout()
    out_path = "attention_visualization.png"
    plt.savefig(out_path)
    print(f"Attention visualization saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--visualize", action="store_true", help="Save an attention heatmap plot")
    args = parser.parse_args()

    device = config.DEVICE
    encoder, decoder, vocab = load_model(device)

    image = Image.open(args.image).convert("RGB")
    image_tensor = eval_transform(image)

    seq, alphas = generate_caption_beam_search(encoder, decoder, image_tensor, vocab, device)
    words = vocab.denumericalize(seq[1:])  # drop leading <SOS>
    caption = " ".join(words)

    print("\nGenerated caption:")
    print(f"  {caption}")

    if args.visualize:
        visualize_attention(args.image, words, alphas[1:len(words) + 1])


if __name__ == "__main__":
    main()
