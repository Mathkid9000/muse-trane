import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from helpers import SEQ_LEN
from model_util import load_model_only
from tokenizer import QuintupleTokenizer, tokens_to_score, load_dataset, detokenize_ids, load_vocab_list
from tune import load_head_lengths, Tune
from generate import generate_score
from similarity_analysis import calculate_frequencies, summarize, plot
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle


GENERATION_CACHE_PATH = Path('./analysis/generated_tunes_cache.pkl')
GENERATION_MAX_TOKENS = 512

def generate_tunes(generations_per_tune=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tune_names, encoded_tunes = load_dataset()
    tokenizer = QuintupleTokenizer()
    head_lengths = load_head_lengths()

    model, _ = load_model_only('bird-v1', device)

    generated_tunes = []

    for name, encoded in tqdm(zip(tune_names, encoded_tunes), desc="Generating tunes", total=len(tune_names)):
        name_reduced = name.split('/')[-1].split('.')[0].replace('_', ' ')
        head_length = head_lengths[name_reduced]
        if head_length == 0: continue

        # bar advance carried by each field-0 token: M_NEXT = +1, M_SKIP_k = +k
        bar_step = {tid: (1 if tok == 'M_NEXT' else int(tok.split('_')[-1]))
                    for tok, tid in tokenizer.vocabs[0].items()
                    if tok == 'M_NEXT' or tok.startswith('M_SKIP_')}
        
        encoded = np.asarray(encoded, dtype=np.int64)
        bar = 0
        idx = 0
        while idx < len(encoded):
            bar += bar_step.get(int(encoded[idx][0]), 0)
            if bar >= head_length:
                break
            idx += 1
        seed = encoded[:idx]

        for _ in range(generations_per_tune):
            encoded_ids = generate_score(model, tokenizer, seed, max_events=len(seed) + GENERATION_MAX_TOKENS, temperature=0.8, top_k=30, top_p=0.9, verbose=False)
            decoded_ids = tokenizer.decode(encoded_ids)
            _, score = tokens_to_score(decoded_ids, export=False)
            tune = Tune(name_reduced)
            tune.load_from_score(score, head_length=head_length)
            generated_tunes.append(tune)
    return generated_tunes


def cached_generated(tunes=[]):
    if GENERATION_CACHE_PATH.exists() and tunes == []:
        with open(GENERATION_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    if tunes != []:
        with open(GENERATION_CACHE_PATH, 'wb') as f:
            pickle.dump(tunes, f)


if __name__ == "__main__":
    generated_tunes = generate_tunes()
    cached_generated(generated_tunes)

    # generated_tunes = cached_generated()
    direct, crossed = calculate_frequencies(generated_tunes)
    summarize(direct, crossed)
    plot(direct, crossed)
    plt.show()