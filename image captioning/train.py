"""
train.py
--------
Trains the EncoderCNN + DecoderRNNWithAttention captioning model end to end.

Usage:
    python train.py

Make sure config.py points at your dataset (IMAGE_DIR / CAPTIONS_FILE) and
that the captions.txt file has two columns named "image" and "caption",
e.g. the standard Flickr8k / Flickr30k format:

    image,caption
    1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up stairs.
    1000268201_693b08cb0e.jpg,A girl going into a wooden building.
    ...
"""

import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from dataset import CaptionDataset, CapCollate, train_transform, eval_transform, build_vocab_from_captions
from models.model import EncoderCNN, DecoderRNNWithAttention
from utils import save_checkpoint, clip_gradients, AverageMeter


def get_dataloaders(vocab):
    full_dataset = CaptionDataset(config.IMAGE_DIR, config.CAPTIONS_FILE, vocab, transform=train_transform)

    val_size = int(len(full_dataset) * config.VALID_SPLIT)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset.transform = eval_transform  # use non-augmented transform for validation

    pad_idx = vocab.stoi["<PAD>"]
    collate_fn = CapCollate(pad_idx)

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
    )
    return train_loader, val_loader


def run_epoch(encoder, decoder, loader, criterion, optimizer, device, training: bool):
    encoder.train(mode=training and config.ENCODER_FINE_TUNE)
    decoder.train(mode=training)
    loss_meter = AverageMeter()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        loop = tqdm(loader, desc="train" if training else "valid")
        for images, captions, lengths in loop:
            images, captions = images.to(device), captions.to(device)
            lengths = lengths.to(device)

            encoder_out = encoder(images)
            predictions, sorted_captions, decode_lengths, alphas, _ = decoder(
                encoder_out, captions, lengths
            )

            # Targets are the captions shifted by one (predict next word).
            targets = sorted_captions[:, 1:]

            # pack_padded_sequence removes the padded time steps so the loss
            # is only computed over real tokens.
            packed_preds = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            packed_targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(packed_preds, packed_targets)
            # Doubly stochastic attention regularization (encourages each pixel
            # to receive roughly equal total attention across all time steps).
            loss += 1.0 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            if training:
                optimizer.zero_grad()
                loss.backward()
                clip_gradients(optimizer, config.GRAD_CLIP)
                optimizer.step()

            loss_meter.update(loss.item(), sum(decode_lengths))
            loop.set_postfix(loss=loss_meter.avg)

    return loss_meter.avg


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print("Building vocabulary...")
    vocab = build_vocab_from_captions(config.CAPTIONS_FILE)
    vocab.save(config.VOCAB_PATH)

    train_loader, val_loader = get_dataloaders(vocab)

    encoder = EncoderCNN(config.ENCODER_TYPE, fine_tune=config.ENCODER_FINE_TUNE).to(device)
    decoder = DecoderRNNWithAttention(
        embed_size=config.EMBED_SIZE,
        hidden_size=config.DECODER_HIDDEN_SIZE,
        vocab_size=len(vocab),
        encoder_dim=encoder.feature_dim,
        attention_dim=config.ATTENTION_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    params = list(decoder.parameters())
    if config.ENCODER_FINE_TUNE:
        encoder.unfreeze_last_block()
        params += list(filter(lambda p: p.requires_grad, encoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        train_loss = run_epoch(encoder, decoder, train_loader, criterion, optimizer, device, training=True)
        val_loss = run_epoch(encoder, decoder, val_loader, criterion, optimizer, device, training=False)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "epoch": epoch,
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "vocab_size": len(vocab),
            }, config.BEST_MODEL_PATH)

    print("Training complete. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
