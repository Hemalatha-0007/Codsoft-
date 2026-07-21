"""
vocabulary.py
-------------
Builds a word <-> index mapping from the training captions and provides
utilities to convert text captions into tensors of token ids and back.
"""

import re
import pickle
from collections import Counter


class Vocabulary:
    """
    Simple whitespace + punctuation tokenizer and vocabulary builder.
    Special tokens:
        <PAD> -> 0   used to pad sequences to equal length in a batch
        <SOS> -> 1   start-of-sequence, prepended to every caption
        <EOS> -> 2   end-of-sequence, appended to every caption
        <UNK> -> 3   any word below the frequency threshold
    """

    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9' ]", " ", text)   # strip punctuation
        return text.split()

    def build_vocabulary(self, sentence_list):
        """Populate stoi/itos from a list of raw caption strings."""
        frequencies = Counter()
        idx = len(self.itos)

        for sentence in sentence_list:
            frequencies.update(self.tokenize(sentence))

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"Vocabulary built: {len(self.itos)} tokens "
              f"(from {len(frequencies)} unique words seen).")

    def numericalize(self, text: str):
        """Convert a raw caption string into a list of token ids (no SOS/EOS)."""
        tokens = self.tokenize(text)
        return [self.stoi.get(tok, self.stoi["<UNK>"]) for tok in tokens]

    def denumericalize(self, indices):
        """Convert a list of token ids back into a list of words, stopping at <EOS>."""
        words = []
        for idx in indices:
            word = self.itos.get(int(idx), "<UNK>")
            if word == "<EOS>":
                break
            if word not in ("<SOS>", "<PAD>"):
                words.append(word)
        return words

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)
