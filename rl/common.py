"""Path setup and the head tasks (seed ids + Omnibook tune) shared by the RL scripts.

Every repo module resolves paths relative to the repo root (env.env, ./storage, ./analysis), so
importing this first makes the rl scripts runnable from any directory."""
import os
import sys
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RL_DIR = Path(__file__).resolve().parent
for p in (ROOT, ROOT / 'analysis', RL_DIR):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
os.chdir(ROOT)

import numpy as np
from tokenizer import QuintupleTokenizer, load_dataset, load_vocab_list, transpose_tok, TRANSPOSE_INTERVALS
from tune import load_head_lengths
from similarity_analysis import base_title
import config

SPLIT_PATH = RL_DIR / 'split.json'


class HeadTask:
    """One Omnibook tune: the head as seed ids, and where its Omnibook head sits in the reward reference."""
    def __init__(self, name, seed, head_length):
        self.name = name                                   # dataset name, eg. 'omnibook/Anthropology.xml'
        self.file = name.split('/')[-1]                    # == Tune.title of the Omnibook tune
        self.base = base_title(self.file)
        self.seed = seed                                   # (n, 5) int64 ids up to the end of the head
        self.head_length = head_length

    def __repr__(self):
        return f'HeadTask({self.file}, {len(self.seed)} seed rows, {self.head_length} bars)'


def head_seed(encoded, head_length, tokenizer):
    """Rows before the bar where the head ends (same rule as analysis/model_similarity.py)."""
    bar_step = { tid: (1 if tok == 'M_NEXT' else int(tok.split('_')[-1]))
                 for tok, tid in tokenizer.vocabs[0].items() if tok == 'M_NEXT' or tok.startswith('M_SKIP_') }
    encoded = np.asarray(encoded, dtype=np.int64)
    bar = 0
    for idx in range(len(encoded)):
        bar += bar_step.get(int(encoded[idx][0]), 0)
        if bar >= head_length: return encoded[:idx]
    return encoded


def load_tasks(tokenizer):
    names, encoded = load_dataset()
    head_lengths = load_head_lengths()
    tasks = []
    for name, enc in zip(names, encoded):
        reduced = name.split('/')[-1].split('.')[0].replace('_', ' ')
        hl = head_lengths.get(reduced, 0)
        if hl == 0: continue
        tasks.append(HeadTask(name, head_seed(enc, hl, tokenizer), hl))
    return tasks


def load_split(tasks):
    """(train, heldout) tasks, grouped by base title so versions of a tune never straddle the split.
    bird-v1's own val tunes are held out first (the base model never trained on them); persisted."""
    if SPLIT_PATH.exists():
        heldout = set(json.loads(SPLIT_PATH.read_text()))
    else:
        from model_util import load_val_split_only
        try: base_val = load_val_split_only(config.BASE_MODEL)
        except FileNotFoundError: base_val = []
        by_base = {}
        for t in tasks: by_base.setdefault(t.base, []).append(t.name)
        chosen = { t.base for t in tasks if t.name in set(base_val) }
        rest = sorted(set(by_base) - chosen)
        random.Random(config.SEED).shuffle(rest)
        while len(chosen) < config.N_HELDOUT and rest: chosen.add(rest.pop())
        heldout = { n for b in chosen for n in by_base[b] }
        SPLIT_PATH.write_text(json.dumps(sorted(heldout), indent=2))
    return [ t for t in tasks if t.name not in heldout ], [ t for t in tasks if t.name in heldout ]


class Transposer:
    """id -> id tables per interval, the same construction dataset.QuintupleDataset uses."""
    def __init__(self):
        vocab = load_vocab_list()
        ids = [ { tok: i for i, tok in enumerate(head) } for head in vocab ]
        self.tables = { iv: [ np.array([ ids[f].get(transpose_tok(tok, iv), i) for i, tok in enumerate(vocab[f]) ], np.int64)
                              for f in range(5) ] for iv in TRANSPOSE_INTERVALS }

    def __call__(self, seed, interval):
        return np.stack([ self.tables[interval][f][seed[:, f]] for f in range(5) ], axis=1)

    def random(self, seed, rng):
        return self(seed, rng.choice(list(self.tables)))
