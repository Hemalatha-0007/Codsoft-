"""
transformer_model.py
---------------------
Alternative decoder: a Transformer decoder instead of an LSTM. The image is
still encoded with the same pretrained CNN (see EncoderCNN in model.py),
producing a grid of spatial features (e.g. 196 x 2048 for ResNet50). Those
features are treated as the "memory" / encoder sequence that the Transformer
decoder cross-attends to, exactly like the encoder output in a text-to-text
Transformer (e.g. "Show, Attend and Tell" reimagined with self-attention
instead of a recurrent cell, in the spirit of Vaswani et al. 2017 /
image-transformer captioning models).

Swap-in usage: reuse EncoderCNN from model.py unchanged, and use
TransformerCaptionDecoder here in place of DecoderRNNWithAttention. Training
loop needs minor changes (causal mask, no teacher-forcing loop over
LSTMCell steps) — see the `forward` docstring below for the expected shapes.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, added to word embeddings
    since the Transformer has no built-in notion of sequence order."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class EncoderFeatureProjection(nn.Module):
    """Projects CNN feature dim (e.g. 2048 for ResNet50) down to the
    Transformer's d_model so it can be used as cross-attention memory."""

    def __init__(self, encoder_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, encoder_out):
        # encoder_out: (batch, num_pixels, encoder_dim) -> (batch, num_pixels, d_model)
        return self.norm(self.proj(encoder_out))


class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1, max_len=50):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.feature_proj = EncoderFeatureProjection(encoder_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def generate_causal_mask(size, device):
        """Prevents each position from attending to future tokens."""
        mask = torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)
        return mask

    def forward(self, encoder_out, captions, pad_idx):
        """
        Teacher-forced training forward pass.
        encoder_out: (batch, num_pixels, encoder_dim)  -- CNN spatial features
        captions:    (batch, seq_len)                  -- token ids incl. <SOS>...<EOS>, padded
        returns:     logits (batch, seq_len-1, vocab_size)
        """
        device = captions.device
        memory = self.feature_proj(encoder_out)              # (B, num_pixels, d_model)

        decoder_input = captions[:, :-1]                     # feed all but last token
        tgt_key_padding_mask = decoder_input == pad_idx       # (B, seq_len-1) True where padded

        tgt_emb = self.embedding(decoder_input) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        causal_mask = self.generate_causal_mask(decoder_input.size(1), device)

        decoded = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.fc_out(decoded)                          # (B, seq_len-1, vocab_size)
        return logits

    @torch.no_grad()
    def generate_greedy(self, encoder_out, sos_idx, eos_idx, device, max_len=None):
        """Greedy (non-beam) decoding for a single image at inference time."""
        max_len = max_len or self.max_len
        memory = self.feature_proj(encoder_out)                # (1, num_pixels, d_model)

        generated = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_emb = self.embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoding(tgt_emb)
            causal_mask = self.generate_causal_mask(generated.size(1), device)

            decoded = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
            next_logits = self.fc_out(decoded[:, -1, :])         # (1, vocab_size)
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_idx:
                break

        return generated.squeeze(0).tolist()


"""
Notes on wiring this into train.py as an alternative to the LSTM decoder:

    from models.model import EncoderCNN
    from models.transformer_model import TransformerCaptionDecoder

    encoder = EncoderCNN(config.ENCODER_TYPE, fine_tune=config.ENCODER_FINE_TUNE)
    decoder = TransformerCaptionDecoder(
        vocab_size=len(vocab), encoder_dim=encoder.feature_dim,
        d_model=512, nhead=8, num_layers=6, max_len=config.MAX_CAPTION_LEN,
    )

    encoder_out = encoder(images)
    logits = decoder(encoder_out, captions, pad_idx=vocab.stoi["<PAD>"])
    loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))

The loss should mask out <PAD> positions, e.g. by constructing the criterion
with `nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])`.
"""
