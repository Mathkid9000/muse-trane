import math
import torch
from helpers import SEQ_LEN, BATCH_SIZE
from tokenizer import load_vocab_list, transpose_tok, TRANSPOSE_INTERVALS
import numpy as np

class QuintupleDataset(torch.utils.data.Dataset):
    """Transpose encoded to every key, concatenate into one (N,5) stream and chunk
    into (x, y) = ((seq_len,5), (seq_len,5)) windows shifted by one event."""

    def __init__(self, encoded, seq_len=SEQ_LEN):
        # scores are stored in their original key only and transposed here instead, so a score and
        # its transpositions can never land on opposite sides of the train/val split. Transposing a
        # row only renames its pitch tokens, so each interval is one id -> id table per field.
        self.seq_len = seq_len
        vocab = load_vocab_list()
        ids = [ { tok: idx for idx, tok in enumerate(head) } for head in vocab ]
        tables = { i: [ np.array([ ids[f].get(transpose_tok(tok, i), idx) for idx, tok in enumerate(vocab[f]) ], dtype=np.int16) for f in range(5) ] for i in TRANSPOSE_INTERVALS }
        segments = [ (s, i) for s in range(len(encoded)) for i in TRANSPOSE_INTERVALS ]
        print(TRANSPOSE_INTERVALS)
        np.random.default_rng(0).shuffle(segments)
        self.encoded = np.concatenate([
            np.stack([ tables[i][f][encoded[s][:, f]] for f in range(5) ], axis=1)
            for s, i in segments
        ], axis=0)

    def __len__(self):
        return max(0, len(self.encoded) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_len + 1]
        # widen at the window boundary, since nn.Embedding and cross_entropy both
        # want int64; one window is 5 * seq_len ids, so the cast costs nothing
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


def make_dataloader(encoded_events, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True):
    dataset = QuintupleDataset(encoded_events, seq_len=seq_len)
    subset = torch.utils.data.Subset(dataset, range(0, len(dataset), seq_len // 4))
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
