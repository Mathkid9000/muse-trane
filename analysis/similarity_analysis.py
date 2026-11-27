import pathlib
import pickle
import re
import time
import numpy as np
import matplotlib.pyplot as plt
import kernels
from tune import Tune, Note, load_tunes

N = 6
POWER = 10
BLOCK = 1024
CACHE_DIR = pathlib.Path("./analysis/tunes_cache.pkl")
# name -> how to build the similarity from (contour, rhythm); combination matches Note.calculate_difference
TYPES = {
    'Contour': lambda c, r: c,
    'Rhythmic': lambda c, r: r,
    'Combination': lambda c, r: c * r,
}

def cached_tunes():
    if CACHE_DIR.exists():
        with open(CACHE_DIR, 'rb') as f: return pickle.load(f)
    tunes = load_tunes()
    with open(CACHE_DIR, 'wb') as f: pickle.dump(tunes, f)
    return tunes

def base_title(title):
    return re.sub(r'_\d+$', '', title.split('.')[0])

def frequency_matrix(tunes, notes_of, index):
    X = np.zeros((len(tunes), len(index)), np.float32)
    for i, tune in enumerate(tunes):
        for gram_id, freq in Tune.count_n_grams(notes_of(tune), N).items():
            X[i, index[gram_id]] = freq
    return X / np.maximum(X.sum(1, keepdims=True), 1)

def leave_one_out_mean(X, same):
    """Row i = mean of the rows of X whose tune is not a version of tune i."""
    excluded = same.astype(np.float32)
    return (X.sum(0) - excluded @ X) / (len(X) - excluded.sum(1, keepdims=True))

def kernel_grams(vocab, D):
    """G[type] = D F D^T with F = S ** POWER, streamed over row blocks of S so F is never stored."""
    contour = Note.encode_grams(vocab, Note.contour_elements)
    rhythm = Note.encode_grams(vocab, Note.rhythm_elements)
    G = { name: np.zeros((len(D), len(D))) for name in TYPES }
    for i0 in range(0, len(vocab), BLOCK):
        i1 = min(i0 + BLOCK, len(vocab))
        c = kernels.similarity_block(*contour, kernels.CONTOUR_W, i0, i1)
        r = kernels.similarity_block(*rhythm, kernels.RHYTHM_W, i0, i1)
        for name, combine in TYPES.items():
            G[name] += D[:, i0:i1] @ (combine(c, r) ** POWER @ D.T)
    return G

def centered_correlation(G, a, b):
    """r = (dh F do) / sqrt((dh F dh)(do F do)) between rows a and b of D (broadcast), given G = D F D^T. nan if not defined."""
    den = G[a, a] * G[b, b]
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(den > 0, G[a, b] / np.sqrt(den), np.nan)
    return np.clip(r, -1, 1)

def plot(direct, crossed):
    fig, axes = plt.subplots(2, len(TYPES), sharex=True, figsize=(4 * len(TYPES), 6))
    bins = np.linspace(-1, 1, 101)
    for col, name in enumerate(TYPES):
        for row, (label, values) in enumerate((('Direct', direct), ('Crossed', crossed))):
            ax, v = axes[row, col], values[name][~np.isnan(values[name])]
            ax.hist(v, bins=bins, color='C0' if row == 0 else 'C1', alpha=0.8)
            ax.axvline(v.mean(), color='k', linestyle='--', linewidth=1)
            ax.text(v.mean(), ax.get_ylim()[1] * 0.92, f' mean {v.mean():.3g}', fontsize=9)
            ax.axvline(0, color='r', linestyle='-', linewidth=0.5)
            ax.set_title(f'{label}: {name}')
            ax.set_ylabel('tunes')
            ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[-1]: ax.set_xlabel('centered kernel correlation')
    fig.suptitle(f'Head vs output similarity ({N}-grams, F = S^{POWER})')
    fig.tight_layout()

def summarize(direct, crossed):
    try: from scipy.stats import wilcoxon
    except ImportError: wilcoxon = None
    print(f'{"type":<12} {"direct":>8} {"crossed":>8} {"p":>10} {"nan d/c":>8}')
    for name in TYPES:
        d, c = direct[name], crossed[name]
        ok = ~np.isnan(d) & ~np.isnan(c)
        p = wilcoxon(d[ok], c[ok]).pvalue if wilcoxon else float('nan')
        print(f'{name:<12} {np.nanmean(d):8.3f} {np.nanmean(c):8.3f} {p:10.3g} {np.isnan(d).sum():>4}/{np.isnan(c).sum()}')

def calculate_frequencies(tunes):
    T = len(tunes)
    vocab = sorted(set(Tune.count_head_n_grams(tunes, N)) | set(Tune.count_output_n_grams(tunes, N)))
    index = { g: i for i, g in enumerate(vocab) }
    print(f'{T} tunes, vocabulary {len(vocab)} {N}-grams')

    bases = np.array([ base_title(t.title) for t in tunes ])
    same = bases[:, None] == bases[None, :]
    H = frequency_matrix(tunes, lambda t: t.head_notes, index)
    O = frequency_matrix(tunes, lambda t: t.output_notes, index)
    D = np.vstack([ H - leave_one_out_mean(H, same), O - leave_one_out_mean(O, same) ])

    t0 = time.time()
    G = kernel_grams(vocab, D)
    print(f'kernel build {time.time() - t0:.1f}s')

    h, o = np.arange(T), T + np.arange(T)
    direct = { name: centered_correlation(g, h, o) for name, g in G.items() }
    crossed = {}
    for name, g in G.items():
        r = centered_correlation(g, h[:, None], o[None, :])
        r[same] = np.nan
        crossed[name] = np.nanmean(r, axis=1)
    return direct, crossed

if __name__ == '__main__':
    tunes = cached_tunes()
    direct, crossed = calculate_frequencies(tunes)
    summarize(direct, crossed)
    plot(direct, crossed)
    plt.show()
