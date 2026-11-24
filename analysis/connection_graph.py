"""Interactive similarity graph of all head n-grams.

Each node is an n-gram (from Tune.count_head_n_grams). Nodes are laid out with
metric MDS so that the distance between two nodes approximates
NGramNote.calculate_diff between them. Node color is frequency. Edges connect
each node to its most similar neighbours; opacity/width shows similarity.
Hovering a node shows the n-gram; the most frequent ones (--render, default 100)
also pop up in music notation (rendered via MuseScore).

Run from the repo root:  python analysis/connection_graph.py --n 8
"""
import argparse
import glob
import hashlib
import json
import pathlib
import subprocess
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import music21
import numpy as np
from matplotlib.collections import LineCollection
from numba import njit, prange
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.sparse.linalg import eigsh
from sklearn.manifold import smacof
from tqdm import tqdm

from melodic_analysis import (OMNIBOOK_DIR, NGramNote, Tune, _build_ngram_score,
                              _find_mscore, _gram_label)

CACHE_DIR = pathlib.Path(__file__).parent / '.ngram_cache'
RENDER_DPI = 150


# --- data ---
def load_tunes_with_progress():
    files = sorted(OMNIBOOK_DIR.iterdir())
    tunes = []
    for file in tqdm(files, desc='Loading tunes', unit='tune'):
        tune = Tune(file.name)
        tune.load_from_file(file)
        tunes.append(tune)
    return tunes


def sorted_grams(n_gram_dict):
    return sorted(n_gram_dict.items(), key=lambda x: x[1], reverse=True)


# numba copy of NGramNote.calculate_diff (weighted edit distance + transposition search).
# Keep in sync with melodic_analysis.py; the constants are passed in from NGramNote.
@njit(cache=True)
def _edit_distance_nb(p1, d1, p2, d2, rest, shift, indel, octave_factor, buf):
    m, k = p1.shape[0], p2.shape[0]
    buf[0, 0] = 0.0
    for i in range(1, m + 1):
        buf[i, 0] = buf[i-1, 0] + indel * min(1.0, 0.5 + d1[i-1])
    for j in range(1, k + 1):
        buf[0, j] = buf[0, j-1] + indel * min(1.0, 0.5 + d2[j-1])
    for i in range(1, m + 1):
        del_cost = indel * min(1.0, 0.5 + d1[i-1])
        r1 = p1[i-1] == rest
        for j in range(1, k + 1):
            ins_cost = indel * min(1.0, 0.5 + d2[j-1])
            r2 = p2[j-1] == rest
            rhythm = min(abs(np.log2(d1[i-1] / d2[j-1])) / 4.0, 0.5)
            if r1 != r2:
                sub = 1.0 + rhythm
            elif r1:
                sub = rhythm
            else:
                diff = p1[i-1] - (p2[j-1] + shift)
                if diff == 0:
                    pc = 0.0
                else:
                    pc = min(abs(diff), 12) / 12.0
                    if diff % 12 == 0:
                        pc *= octave_factor
                sub = pc + rhythm
            buf[i, j] = min(buf[i-1, j] + del_cost,
                            buf[i, j-1] + ins_cost,
                            buf[i-1, j-1] + sub)
    return buf[m, k]


@njit(parallel=True, cache=True)
def _diff_rows_nb(P, Dur, start, end, rest, max_shift, indel, octave_factor, out):
    N, n = P.shape
    for r in prange(end - start):
        i = start + r
        buf = np.empty((n + 1, n + 1))
        for j in range(i + 1, N):
            best = np.inf
            for s in range(-max_shift, max_shift + 1):
                d = _edit_distance_nb(P[i], Dur[i], P[j], Dur[j], rest, s,
                                      indel, octave_factor, buf)
                if d < best:
                    best = d
            out[r, j] = best


def diff_matrix(gram_ids, chunk=64):
    """Pairwise NGramNote.calculate_diff for every pair (numba-compiled, parallel)."""
    notes = [NGramNote.gram_id_to_notes(g) for g in gram_ids]
    P = np.array([[x.pitch for x in ns] for ns in notes], dtype=np.int64)
    Dur = np.array([[float(x.dur) for x in ns] for ns in notes], dtype=np.float64)
    N = len(P)
    D = np.zeros((N, N))
    args = (NGramNote.REST, NGramNote.DIFF_MAX_SHIFT, NGramNote.DIFF_INDEL,
            NGramNote.DIFF_OCTAVE_FACTOR)
    print('Compiling diff kernel (first run only)...')
    _diff_rows_nb(P[:2], Dur[:2], 0, 1, *args, np.zeros((1, 2)))  # trigger JIT before the bar starts
    with tqdm(total=N * (N - 1) // 2, desc='Pairwise diffs', unit='pair', unit_scale=True) as bar:
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            out = np.zeros((e - s, N))
            _diff_rows_nb(P, Dur, s, e, *args, out)
            D[s:e] = out
            bar.update(sum(N - 1 - i for i in range(s, e)))
    D = np.triu(D, 1)
    return D + D.T


def layout(D, refine_rounds=10, iters_per_round=10):
    """2D positions whose Euclidean distances approximate D.

    Classical MDS (fast eigen-decomposition) gives the starting layout, then a few
    rounds of SMACOF (metric MDS stress minimisation) refine it.
    """
    print('Computing layout...')
    N = len(D)
    J = np.eye(N) - 1.0 / N
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = eigsh(B, k=2, which='LA')
    X = vecs * np.sqrt(np.maximum(vals, 1e-12))
    for _ in tqdm(range(refine_rounds), desc='Refining layout', unit='round'):
        X, _ = smacof(D, n_components=2, init=X, n_init=1, max_iter=iters_per_round,
                      normalized_stress='auto')
    return X


def knn_edges(D, k):
    """Each node's k most similar neighbours, as unique (i, j) pairs."""
    N = len(D)
    k = min(k, N - 1)
    Dm = D + np.diag(np.full(N, np.inf))
    nbrs = np.argpartition(Dm, k, axis=1)[:, :k]
    pairs = {(min(i, j), max(i, j)) for i in range(N) for j in nbrs[i]}
    return np.array(sorted(pairs))


# --- notation rendering ---
def _crop_whitespace(img, pad=10):
    rgb = img[..., :3]
    alpha = img[..., 3] if img.shape[-1] == 4 else np.ones(img.shape[:2])
    ink = (rgb.min(axis=-1) < 0.9) & (alpha > 0.1)
    rows, cols = np.where(ink)
    if rows.size == 0:
        return img
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, img.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, img.shape[1])
    out = img[r0:r1, c0:c1].copy()
    if out.shape[-1] == 4:  # flatten transparency onto white
        a = out[..., 3:4]
        out = np.concatenate([out[..., :3] * a + (1 - a), np.ones_like(a)], axis=-1)
    return out


def render_notation(grams):
    """Return {gram_id: cropped image}. Renders missing grams in one MuseScore batch job."""
    CACHE_DIR.mkdir(exist_ok=True)

    def key(gram_id, freq):
        return hashlib.md5(f'{gram_id}#{freq}'.encode()).hexdigest()[:16]

    def png_for(k):
        hits = sorted(glob.glob(str(CACHE_DIR / f'{k}*.png')))
        return hits[0] if hits else None

    todo = [(g, f) for g, f in grams if not png_for(key(g, f))]
    jobs = []
    for gram_id, freq in tqdm(todo, desc='Building scores', unit='gram', disable=not todo):
        k = key(gram_id, freq)
        xml_fp = CACHE_DIR / f'{k}.musicxml'
        score = _build_ngram_score([(gram_id, freq)])
        score.metadata = music21.metadata.Metadata(title=' ', composer=' ')  # no header text
        score.write('musicxml', fp=str(xml_fp))
        jobs.append((k, {'in': str(xml_fp), 'out': str(CACHE_DIR / f'{k}.png')}))

    if jobs:
        job_fp = CACHE_DIR / 'job.json'
        job_fp.write_text(json.dumps([j for _, j in jobs]))
        proc = subprocess.Popen([_find_mscore(), '-j', str(job_fp), '-r', str(RENDER_DPI)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # MuseScore gives no progress output, so count the PNGs as they appear
        with tqdm(total=len(jobs), desc='Rendering notation (MuseScore)', unit='gram') as bar:
            while True:
                done = sum(1 for k, _ in jobs if png_for(k))
                bar.update(done - bar.n)
                if proc.poll() is not None:
                    break
                time.sleep(0.25)

    images = {}
    for gram_id, freq in tqdm(grams, desc='Loading notation images', unit='gram'):
        fp = png_for(key(gram_id, freq))
        if fp is None:
            tqdm.write(f'warning: no notation rendered for {gram_id}')
            continue
        images[gram_id] = _crop_whitespace(mpimg.imread(fp))
    return images


# --- plot ---
def draw_graph(grams, D, pos, images, edges):
    freqs = np.array([f for _, f in grams], dtype=float)
    labels = [_gram_label(g) for g, _ in grams]
    N = len(grams)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(f'{N} head n-grams ({len(images)} with notation, outlined) — '
                 'distance ≈ calculate_diff, color = frequency\n(hover a node to see it)')
    ax.set_aspect('equal')
    ax.axis('off')

    # Edges: similarity = 1 - normalized diff
    max_d = D.max() if D.max() > 0 else 1.0
    sims = 1 - D[edges[:, 0], edges[:, 1]] / max_d
    edge_colors = np.zeros((len(sims), 4))
    edge_colors[:, :3] = 0.35
    edge_colors[:, 3] = 0.04 + 0.4 * sims ** 3
    ax.add_collection(LineCollection(pos[edges], colors=edge_colors,
                                     linewidths=0.2 + 1.5 * sims ** 3, zorder=1))

    # Nodes: least frequent drawn first so frequent ones sit on top
    order = np.argsort(freqs)
    sizes = 4 + 220 * (np.log(freqs) / max(np.log(freqs.max()), 1e-9)) ** 2
    has_img = np.array([g in images for g, _ in grams])
    norm = LogNorm(vmin=freqs.min(), vmax=max(freqs.max(), freqs.min() + 1))
    nodes = ax.scatter(pos[order, 0], pos[order, 1], c=freqs[order], s=sizes[order],
                       cmap='viridis', norm=norm, zorder=2,
                       edgecolors=np.where(has_img[order], 'black', 'none'), linewidths=0.8)
    cbar = fig.colorbar(nodes, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Frequency (log scale)')
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    cbar.ax.yaxis.set_minor_formatter(ScalarFormatter())

    # Hover popup: notation image + text
    popup = AnnotationBbox(OffsetImage(np.ones((2, 2, 3)), zoom=0.5), (0, 0),
                           xybox=(0, 90), xycoords=ax.transData, boxcoords='offset points',
                           pad=0.4, bboxprops=dict(edgecolor='#888', facecolor='white'),
                           zorder=10)
    popup.set_visible(False)
    popup.set_annotation_clip(False)
    fig.add_artist(popup)  # figure-level so it draws over the colorbar
    info = ax.annotate('', (0, 0), xytext=(0, -22), textcoords='offset points',
                       ha='center', va='top', fontsize=8, fontfamily='monospace',
                       bbox=dict(boxstyle='round', fc='white', ec='#888'), zorder=11)
    info.set_visible(False)
    current = [None]

    def on_move(event):
        hit, details = nodes.contains(event) if event.inaxes == ax else (False, {})
        idx = None
        if hit:
            # among hit nodes, prefer the most frequent (drawn on top), then the closest
            cand = order[details['ind']]
            disp = ax.transData.transform(pos[cand])
            dist = np.hypot(*(disp - [event.x, event.y]).T)
            idx = cand[np.lexsort((dist, -freqs[cand]))[0]]
        if idx == current[0]:
            return
        current[0] = idx
        if idx is None:
            popup.set_visible(False)
            info.set_visible(False)
        else:
            gram_id, freq = grams[idx]
            img = images.get(gram_id)
            # place popup above or below depending on where the node is on screen
            _, y_frac = ax.transAxes.inverted().transform(ax.transData.transform(pos[idx]))
            dy = -110 if y_frac > 0.6 else 110
            if img is not None:
                popup.offsetbox.set_data(img)
                popup.offsetbox.set_zoom(min(0.6, 550 / img.shape[1]))
                popup.xy = pos[idx]
                popup.xybox = (0, dy)
                popup.set_visible(True)
            else:
                popup.set_visible(False)
            info.xy = pos[idx]
            info.set_text(f'{labels[idx]}\n×{freq}' + ('' if img is not None else '  (notation not rendered)'))
            info.set_position((0, 22 if dy < 0 else -22))
            info.set_va('bottom' if dy < 0 else 'top')
            info.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    pad = 0.05 * np.ptp(pos, axis=0).max()
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=int, default=5, help='n-gram length')
    parser.add_argument('--render', type=int, default=100,
                        help='render notation for this many of the most frequent n-grams')
    parser.add_argument('--neighbors', type=int, default=5,
                        help='connect each node to its k most similar n-grams')
    args = parser.parse_args()

    tunes = load_tunes_with_progress()
    grams = sorted_grams(Tune.count_head_n_grams(tunes, args.n))
    print(f'{len(grams)} distinct {args.n}-grams')
    if len(grams) < 2:
        print('Not enough n-grams to build a graph.')
        return
    D = diff_matrix([g for g, _ in grams])
    pos = layout(D)
    edges = knn_edges(D, args.neighbors)
    images = render_notation(grams[:args.render])
    print('Opening graph window...')
    draw_graph(grams, D, pos, images, edges)


if __name__ == '__main__':
    main()
