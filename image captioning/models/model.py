""
model.py
--------
Two-part architecture:

1. EncoderCNN
   Wraps a pretrained ResNet50 (or VGG16) from torchvision, strips the final
   classification layer, and keeps the last convolutional feature map. This
   gives a grid of spatial feature vectors (e.g. 14x14x2048 for ResNet50)
   instead of a single flat vector, which lets the decoder attend to
   different regions of the image at each generation step.

2. DecoderRNNWithAttention
   An LSTM that, at every time step, uses Bahdanau-style additive attention
   to compute a weighted sum of the encoder's spatial features, conditioned
   on the decoder's previous hidden state. This weighted context vector is
   concatenated with the current word embedding and fed into the LSTM cell.
   This is the "Show, Attend and Tell" architecture (Xu et al., 2015).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, encoder_type: str = "resnet50", fine_tune: bool = False):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Drop avgpool + fc, keep everything up to the last conv block.
            modules = list(resnet.children())[:-2]
            self.backbone = nn.Sequential(*modules)
            self.feature_dim = 2048

        elif encoder_type == "vgg16":
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.backbone = vgg.features   # last conv feature map, no pooling/fc
            self.feature_dim = 512

        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # By default freeze the pretrained CNN; only unfreeze if fine-tuning.
        for param in self.backbone.parameters():
            param.requires_grad = fine_tune

        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        """
        images: (batch, 3, 224, 224)
        returns: (batch, num_pixels, feature_dim)  e.g. (batch, 196, 2048)
        """
        features = self.backbone(images)                  # (B, C, H, W)
        features = self.adaptive_pool(features)            # (B, C, 14, 14)
        features = features.permute(0, 2, 3, 1)             # (B, 14, 14, C)
        features = features.reshape(features.size(0), -1, features.size(-1))  # (B, 196, C)
        return features

    def unfreeze_last_block(self):
        """Optionally call this to fine-tune the deepest CNN layers only."""
        children = list(self.backbone.children())
        for layer in children[-2:]:
            for param in layer.parameters():
                param.requires_grad = True


class Attention(nn.Module):
    """Additive (Bahdanau) attention over encoder spatial locations."""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out:    (batch, num_pixels, encoder_dim)
        decoder_hidden: (batch, decoder_dim)
        returns: context (batch, encoder_dim), alpha (batch, num_pixels)
        """
        att1 = self.encoder_att(encoder_out)                 # (B, num_pixels, attn_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B, 1, attn_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (B, num_pixels)
        alpha = self.softmax(att)                             # (B, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha


class DecoderRNNWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim,
                 attention_dim, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_dim = encoder_dim

        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        # LSTMCell used so we can run one time-step at a time (needed for attention).
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)

        # Initialize hidden/cell state from the mean of encoder features.
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

        # Gating scalar (from the original paper) that scales the context vector.
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        """
        Teacher-forced training forward pass.
        encoder_out: (batch, num_pixels, encoder_dim)
        captions:    (batch, max_len)  token ids, includes <SOS> ... <EOS>
        lengths:     (batch,)          true length of each caption (with SOS/EOS)
        returns: predictions (batch, max_len-1, vocab_size), sorted captions, decode lengths, alphas
        """
        batch_size = encoder_out.size(0)

        # Sort batch by decreasing caption length (needed for efficient looping).
        lengths, sort_idx = lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        captions = captions[sort_idx]

        embeddings = self.embedding(captions)          # (B, max_len, embed_size)
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (lengths - 1).tolist()          # we predict word t+1 from word t
        max_decode_len = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, encoder_out.size(1)).to(encoder_out.device)

        for t in range(max_decode_len):
            batch_size_t = sum([l > t for l in decode_lengths])
            context, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context = gate * context

            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], context], dim=1)
            h_t, c_t = self.lstm_cell(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            h = torch.cat([h_t, h[batch_size_t:]], dim=0) if batch_size_t < batch_size else h_t
            c = torch.cat([c_t, c[batch_size_t:]], dim=0) if batch_size_t < batch_size else c_t

            preds = self.fc(self.dropout(h_t))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions, decode_lengths, alphas, sort_idx
