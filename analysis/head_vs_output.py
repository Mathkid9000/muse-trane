"""Does a tune's head n-gram distribution shape its solo (output) n-gram distribution?

1. Vocabulary = every n-gram in every tune's head AND output.
2. Lay the whole vocabulary out in 2D by similarity (NGramNote.calculate_diff), using
   landmark MDS, then spread it evenly over the unit square (rank transform per axis)
   so every grid cell holds roughly the same number of vocabulary n-grams. Cached.
3. Per tune: put its head n-grams on that plane weighted by frequency, bin to a
   GRID x GRID array, normalize to a distribution and smooth. Same for its output.
   difference = output - head.
4. Control: for each tune's output grid, rank its own head among all tunes' heads by
   correlation. If heads matter, the own head should rank near the top.

Display (one figure): a row per selected tune with its head grid, output grid and
difference grid, plus one hover map averaging the selected tunes' n-grams, with
notation popups.

Run from the repo root:  python analysis/head_vs_output.py --n 5 --tunes 5
"""
import argparse
import hashlib
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr, wilcoxon
from tqdm import tqdm

from connection_graph import (CACHE_DIR, _edit_distance_nb, attach_notation_hover,
                              render_notation)
from melodic_analysis import NGramNote, _gram_label, load_tunes

LAYOUT_VERSION = 2  # bump to invalidate cached layouts after changing the algorithm


# --- counting (mirrors Tune._count_head_n_grams, but for any note list) ---
def count_n_grams(notes, n):
    counts = {}
    for i in range(len(notes) - n):
        gram_id = NGramNote.notes_to_gram_id(notes[i:i+n])
        if gram_id:
            counts[gram_id] = counts.get(gram_id, 0) + 1
    return counts


# --- distance kernels (numba) ---
@njit(cache=True)
def _calc_diff_nb(p1, d1, p2, d2, rest, max_shift, indel, octave_factor, buf):
    best = np.inf
    for s in range(-max_shift, max_shift + 1):
        d = _edit_distance_nb(p1, d1, p2, d2, rest, s, indel, octave_factor, buf)
        if d < best:
            best = d
    return best


@njit(parallel=True, cache=True)
def _one_to_all_nb(P, Dur, i, rest, max_shift, indel, octave_factor, out):
    N, n = P.shape
    n_chunks = 64
    step = (N + n_chunks - 1) // n_chunks
    for c in prange(n_chunks):
        buf = np.empty((n + 1, n + 1))
        for j in range(c * step, min((c + 1) * step, N)):
            out[j] = _calc_diff_nb(P[i], Dur[i], P[j], Dur[j], rest, max_shift,
                                   indel, octave_factor, buf)


@njit(parallel=True, cache=True)
def _pairs_nb(P, Dur, I, J, rest, max_shift, indel, octave_factor, out):
    n = P.shape[1]
    for k in prange(len(I)):
        buf = np.empty((n + 1, n + 1))
        out[k] = _calc_diff_nb(P[I[k]], Dur[I[k]], P[J[k]], Dur[J[k]], rest, max_shift,
                               indel, octave_factor, buf)


def _encode(gram_ids):
    notes = [NGramNote.gram_id_to_notes(g) for g in gram_ids]
    P = np.array([[x.pitch for x in ns] for ns in notes], dtype=np.int64)
    Dur = np.array([[float(x.dur) for x in ns] for ns in notes], dtype=np.float64)
    return P, Dur


DIFF_ARGS = (NGramNote.REST, NGramNote.DIFF_MAX_SHIFT, NGramNote.DIFF_INDEL,
             NGramNote.DIFF_OCTAVE_FACTOR)


# --- layout ---
def landmark_mds(P, Dur, n_landmarks, seed=0):
    """Landmark MDS (de Silva & Tenenbaum). Returns (positions (N,2), landmark indices).

    Half the landmarks are chosen by farthest-point sampling (covers the extremes),
    the rest at random (covers the dense middle).
    """
    N = len(P)
    k = min(n_landmarks, N)
    rng = np.random.default_rng(seed)
    rows = np.empty((k, N), dtype=np.float32)
    row = np.empty(N)
    chosen = []
    min_dist = np.full(N, np.inf)
    n_fps = k // 2
    print('Compiling diff kernel (first run only)...')
    _one_to_all_nb(P[:2], Dur[:2], 0, *DIFF_ARGS, np.empty(2))
    nxt = int(rng.integers(N))
    for r in tqdm(range(k), desc='Landmark distances', unit='landmark'):
        chosen.append(nxt)
        _one_to_all_nb(P, Dur, nxt, *DIFF_ARGS, row)
        rows[r] = row
        min_dist = np.minimum(min_dist, row)
        if r + 1 < n_fps:
            nxt = int(np.argmax(min_dist))
        elif r + 1 < k:
            remaining = np.setdiff1d(np.arange(N), chosen)
            nxt = int(rng.choice(remaining))
    chosen = np.array(chosen)

    print('Computing layout (landmark MDS)...')
    rows = rows.astype(np.float64)
    D2_LL = rows[:, chosen] ** 2
    D2_LL = (D2_LL + D2_LL.T) / 2
    J = np.eye(k) - 1.0 / k
    B = -0.5 * J @ D2_LL @ J
    vals, vecs = np.linalg.eigh(B)
    top = np.argsort(vals)[::-1][:2]
    vals, vecs = np.maximum(vals[top], 1e-12), vecs[:, top]
    pinv = vecs / np.sqrt(vals)              # (k, 2)
    mu = D2_LL.mean(axis=0)                  # (k,)
    X = -0.5 * ((rows.T ** 2) - mu) @ pinv   # (N, 2): triangulate every point
    return X, chosen


def spread_evenly(X):
    """Map positions onto the unit square so n-grams cover it uniformly.

    Rotate to principal axes, cut into ~sqrt(N) vertical strips holding equal numbers
    of points (by x rank), then spread each strip's points evenly in y (by y rank).
    Neighbours stay neighbours, but every region of the square gets the same share of
    the vocabulary (a per-axis rank transform alone leaves holes, e.g. a ring stays a ring).
    """
    N = len(X)
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xr = Xc @ Vt.T
    out = np.empty_like(Xr)
    n_strips = max(1, int(round(np.sqrt(N))))
    order = np.argsort(Xr[:, 0], kind='stable')
    out[order, 0] = (np.arange(N) + 0.5) / N
    for strip in np.array_split(order, n_strips):
        ys = strip[np.argsort(Xr[strip, 1], kind='stable')]
        out[ys, 1] = (np.arange(len(ys)) + 0.5) / len(ys)
    return out


def layout_quality(P, Dur, X_raw, X_even, n_pairs=5000, seed=1):
    rng = np.random.default_rng(seed)
    I = rng.integers(len(P), size=n_pairs)
    J = rng.integers(len(P), size=n_pairs)
    true = np.empty(n_pairs)
    _pairs_nb(P, Dur, I, J, *DIFF_ARGS, true)
    raw = np.hypot(*(X_raw[I] - X_raw[J]).T)
    even = np.hypot(*(X_even[I] - X_even[J]).T)
    return pearsonr(true, raw)[0], spearmanr(true, even)[0]


def get_layout(vocab, n, n_landmarks):
    """Positions (N, 2) on the evenly spread unit square; the MDS layout is cached."""
    h = hashlib.md5()
    h.update(f'v{LAYOUT_VERSION}|n{n}|L{n_landmarks}|{DIFF_ARGS}|'.encode())
    h.update('\n'.join(vocab).encode())
    cache_fp = CACHE_DIR / f'layout_{h.hexdigest()[:16]}.npz'
    if cache_fp.exists():
        print(f'Using cached layout {cache_fp.name}')
        X_raw = np.load(cache_fp, allow_pickle=False)['X_raw']
    else:
        P, Dur = _encode(vocab)
        X_raw, _ = landmark_mds(P, Dur, n_landmarks)
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez_compressed(cache_fp, X_raw=X_raw)
    pos = spread_evenly(X_raw)  # cheap, so not cached: can be changed freely
    print('Checking layout quality on random pairs...')
    q_raw, q_even = layout_quality(*_encode(vocab), X_raw, pos)
    return pos, q_raw, q_even


# --- grids ---
def to_grid(counts, index, pos, grid, sigma):
    """Frequency-weighted 2D histogram of n-grams, normalized to sum 1, then smoothed."""
    if not counts:
        return np.zeros((grid, grid))
    idx = np.array([index[g] for g in counts])
    w = np.array(list(counts.values()), dtype=float)
    # histogram2d returns [x, y]; transpose so rows are y (imshow convention)
    H, _, _ = np.histogram2d(pos[idx, 0], pos[idx, 1], bins=grid,
                             range=[[0, 1], [0, 1]], weights=w)
    H = H.T / w.sum()
    if sigma > 0:
        H = gaussian_filter(H, sigma, mode='constant')
        H /= H.sum()
    return H


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


# --- display ---
def _short(title):
    return title.removesuffix('.xml').replace('_', ' ')


def show_all(tunes, sel, heads, outs, H, O, ranks, C, pos, index, images, grid):
    """One figure: a row of small grids per tune (head | output | output-head), plus a
    single hover map on the right with every displayed tune's n-grams averaged."""
    k = len(sel)
    fig = plt.figure(figsize=(18, max(8.5, 2.0 * k + 2.0)), layout='constrained')
    gs = fig.add_gridspec(k, 4, width_ratios=[1, 1, 1, 4.6])
    fig.suptitle('Head vs. output n-gram distribution  —  rank = where the own head ranks among all '
                 f'{len(tunes)} heads at predicting the output (1 = best)', fontsize=11)

    # shared color scales across all rows so tunes are comparable
    vmax = max(max(H[i].max(), O[i].max()) for i in sel)
    diffs = {i: O[i] - H[i] for i in sel}
    m = max(max(abs(d).max() for d in diffs.values()), 1e-12)
    dist_norm = PowerNorm(0.5, vmin=0, vmax=vmax)  # sqrt scale keeps faint regions visible
    diff_norm = TwoSlopeNorm(0, -m, m)
    extent = [0, 1, 0, 1]

    grid_axes = {}  # axes -> (tune index, kind)
    col_axes = [[], [], []]
    for r, i in enumerate(sel):
        for c, (G, cmap, norm, kind) in enumerate(((H[i], 'magma', dist_norm, 'head'),
                                                   (O[i], 'magma', dist_norm, 'output'),
                                                   (diffs[i], 'RdBu_r', diff_norm, 'diff'))):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(G, origin='lower', extent=extent, cmap=cmap, norm=norm,
                           interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            grid_axes[ax] = (i, kind)
            col_axes[c].append((ax, im))
        col_axes[0][-1][0].set_ylabel(
            f'{_short(tunes[i].title)}\nrank {ranks[i]}/{len(tunes)}, r={C[i, i]:.2f}', fontsize=9)
    for c, title in enumerate(('Head', 'Output', 'Output − Head')):
        col_axes[c][0][0].set_title(title, fontsize=10)
    cb = fig.colorbar(col_axes[0][0][1], ax=[a for a, _ in col_axes[0] + col_axes[1]],
                      location='bottom', shrink=0.8, aspect=40, label='share of n-grams (sqrt scale)')
    cb.set_ticks(vmax * np.array([0, 1, 4, 9, 16]) / 16)  # evenly spaced on the sqrt scale
    cb.ax.xaxis.set_major_formatter('{x:.4f}')
    cb = fig.colorbar(col_axes[2][0][1], ax=[a for a, _ in col_axes[2]], location='bottom',
                      shrink=0.9, aspect=20, label='red: more in solo · blue: more in head')
    cb.set_ticks([-m, 0, m])
    cb.ax.xaxis.set_major_formatter('{x:+.4f}')

    # --- single hover map: average over the displayed tunes ---
    ax_m = fig.add_subplot(gs[:, 3])
    items = []  # (kind, gram_id, total count, number of tunes containing it)
    for kind, counts in (('head', heads), ('output', outs)):
        total, present = {}, {}
        for i in sel:
            for g, f in counts[i].items():
                total[g] = total.get(g, 0) + f
                present[g] = present.get(g, 0) + 1
        items += [(kind, g, total[g], present[g]) for g in total]
    ipos = np.array([pos[index[g]] for _, g, _, _ in items])
    avg = np.array([t / k for _, _, t, _ in items])
    sizes = 6 + 140 * (avg / avg.max()) ** 0.7
    is_head = np.array([kind == 'head' for kind, *_ in items])
    hi, oi = np.where(is_head)[0], np.where(~is_head)[0]
    oi, hi = oi[np.argsort(avg[oi])], hi[np.argsort(avg[hi])]  # frequent on top
    sc_o = ax_m.scatter(*ipos[oi].T, s=sizes[oi], marker='^', c='#e8743b', alpha=0.55,
                        edgecolors='none', label='output', zorder=2)
    sc_h = ax_m.scatter(*ipos[hi].T, s=sizes[hi], marker='o', c='#2a78d6', alpha=0.8,
                        edgecolors='white', linewidths=0.5, label='head', zorder=3)
    ax_m.legend(loc='upper right', fontsize=8)
    ax_m.set_title(f'Average of the {k} tunes (size = mean count per tune) — hover for notation',
                   fontsize=10)
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 1)
    ax_m.set_aspect('equal')
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    texts = [f'{kind}: {_gram_label(g)}\navg ×{t / k:.2f} per tune (total {t}, in {n}/{k} tunes)'
             for kind, g, t, n in items]
    imgs = [images.get((g, t)) for _, g, t, _ in items]
    fig._hover = attach_notation_hover(fig, ax_m, [(sc_h, hi), (sc_o, oi)], ipos,
                                       avg + is_head * 1e-3, imgs, texts)

    # Hovering a grid cell lists that tune's n-grams in the cell (shown under the map)
    status = ax_m.text(0.0, -0.02, 'Hover a grid cell to list its n-grams',
                       transform=ax_m.transAxes, ha='left', va='top', fontsize=8.5,
                       fontfamily='monospace')
    cells = {}
    for i in sel:
        for kind, counts in (('head', heads[i]), ('output', outs[i])):
            for g, f in counts.items():
                cx, cy = np.minimum((pos[index[g]] * grid).astype(int), grid - 1)
                cells.setdefault((i, kind, cx, cy), []).append((f, g))

    def on_cell(event):
        if event.inaxes not in grid_axes or event.xdata is None:
            return
        i, _ = grid_axes[event.inaxes]
        cx = min(int(event.xdata * grid), grid - 1)
        cy = min(int(event.ydata * grid), grid - 1)

        def top(kind):
            here = sorted(cells.get((i, kind, cx, cy), []), reverse=True)[:3]
            return ', '.join(f'{_gram_label(g)} ×{f}' for f, g in here) or '–'
        status.set_text(f'{_short(tunes[i].title)}  cell ({cx},{cy})  head {H[i][cy, cx]:.4f}  '
                        f'output {O[i][cy, cx]:.4f}  diff {diffs[i][cy, cx]:+.4f}\n'
                        f'head:   {top("head")}\noutput: {top("output")}')
        fig.canvas.draw_idle()

    fig._cell_hover = fig.canvas.mpl_connect('motion_notify_event', on_cell)
    return fig


def grid_similarity(head_grid, out_grid):
    """1 - (sum over pixels of |output - head|) / 2.

    Both grids sum to 1, so the pixel-summed absolute difference runs from 0 (identical)
    to 2 (no overlap); this maps it to a similarity from 1 (identical) to 0.
    """
    return 1 - np.abs(out_grid - head_grid).sum() / 2


def show_similarity_histograms(tunes, H, O):
    """Top: each tune's head vs. its own output. Bottom: each tune's output vs. the other
    tunes' heads, averaged per tune. Same bins and axes so the two can be compared."""
    N = len(tunes)
    S = np.array([[grid_similarity(H[i], O[j]) for i in range(N)] for j in range(N)])  # [output, head]
    same = np.diag(S).copy()
    diff = np.array([np.delete(S[j], j).mean() for j in range(N)])

    stat = wilcoxon(same, diff)  # paired per tune: own head vs. average other head
    print(f'\nHead/output similarity (1 - sum|output - head| / 2):')
    print(f'  same tune:       mean {same.mean():.3f}  median {np.median(same):.3f}')
    print(f'  different tunes: mean {diff.mean():.3f}  median {np.median(diff):.3f}')
    print(f'  own head more similar for {(same > diff).sum()}/{N} tunes; '
          f'Wilcoxon signed-rank p = {stat.pvalue:.2g}')

    lo, hi = min(same.min(), diff.min()), max(same.max(), diff.max())
    pad = 0.05 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 26)
    fig, (ax_s, ax_d) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, sharey=True,
                                     layout='constrained')
    fig.suptitle(f'Head vs. output similarity, all {N} tunes\n'
                 f'(1 − Σ|output − head| / 2 over grid pixels;  own head more similar for '
                 f'{(same > diff).sum()}/{N} tunes, Wilcoxon p = {stat.pvalue:.2g})', fontsize=11)
    for ax, vals, color, title in (
            (ax_s, same, '#2a78d6', 'Same tune: head vs. its own output'),
            (ax_d, diff, '#e8743b', 'Different tunes: output vs. other tunes\' heads (averaged per tune)')):
        ax.hist(vals, bins=bins, color=color, edgecolor='white')
        ax.axvline(vals.mean(), color='#333', linestyle='--', linewidth=1)
        ax.text(vals.mean(), 0.97, f' mean {vals.mean():.3f}', transform=ax.get_xaxis_transform(),
                va='top', fontsize=9)
        ax.set_title(title, fontsize=10, loc='left')
        ax.set_ylabel('tunes')
        ax.spines[['top', 'right']].set_visible(False)
    ax_d.set_xlabel('similarity (0 = no overlap, 1 = identical distributions)')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=int, default=6, help='n-gram length')
    parser.add_argument('--tunes', type=int, default=20, help='number of tunes to display')
    parser.add_argument('--names', nargs='*', help='tunes to display (substring of file name)')
    parser.add_argument('--grid', type=int, default=64, help='grid resolution (cells per side)')
    parser.add_argument('--sigma', type=float, default=2.0, help='smoothing, in grid cells')
    parser.add_argument('--landmarks', type=int, default=2000, help='landmarks for MDS')
    parser.add_argument('--render', type=int, default=100,
                        help='render notation for the top N head and top N output n-grams on the map')
    parser.add_argument('--seed', type=int, default=0, help='seed for picking tunes')
    parser.add_argument('--save', type=pathlib.Path, help='also save the figure as PNG in this folder')
    parser.add_argument('--no-show', action='store_true', help="don't open windows")
    args = parser.parse_args()

    tunes = load_tunes()
    heads = [count_n_grams(t.head_notes, args.n) for t in tunes]
    outs = [count_n_grams(t.output_notes, args.n) for t in tunes]
    vocab = sorted(set().union(*heads, *outs))
    print(f'{len(vocab)} distinct {args.n}-grams across all heads and outputs')
    index = {g: i for i, g in enumerate(vocab)}

    pos, q_raw, q_even = get_layout(vocab, args.n, args.landmarks)
    print(f'Layout quality: Pearson r(calculate_diff, MDS distance) = {q_raw:.3f}; '
          f'Spearman r(calculate_diff, spread-out distance) = {q_even:.3f}')

    grid_args = (index, pos, args.grid, args.sigma)
    H = [to_grid(h, *grid_args) for h in tqdm(heads, desc='Head grids', unit='tune')]
    O = [to_grid(o, *grid_args) for o in tqdm(outs, desc='Output grids', unit='tune')]

    # Control: for each output, rank every head by correlation; where does the own head land?
    C = np.array([[corr(H[i], O[j]) for i in range(len(tunes))] for j in range(len(tunes))])
    ranks = [1 + int((C[j] > C[j, j]).sum()) for j in range(len(tunes))]
    print(f'\nOwn head rank among {len(tunes)} heads (1 = own head matches the output best):')
    for t, r, j in sorted(zip(tunes, ranks, range(len(tunes))), key=lambda x: x[1]):
        print(f'  {t.title.removesuffix(".xml"):28s} rank {r:2d}   corr {C[j, j]:.3f}   '
              f'(mean corr with other heads {np.delete(C[j], j).mean():.3f})')
    print(f'Mean rank {np.mean(ranks):.1f} (chance: {(len(tunes) + 1) / 2:.1f}); '
          f'own head is #1 for {sum(r == 1 for r in ranks)}/{len(tunes)} tunes')

    if args.names:
        sel = [i for i, t in enumerate(tunes)
               if any(s.lower() in t.title.lower() for s in args.names)]
    else:
        sel = sorted(random.Random(args.seed).sample(range(len(tunes)), min(args.tunes, len(tunes))))
    print('\nDisplaying:', ', '.join(tunes[i].title.removesuffix('.xml') for i in sel))

    # notation for the map: top N head and top N output n-grams by total over the shown tunes
    to_render = []
    for counts in (heads, outs):
        total = {}
        for i in sel:
            for g, f in counts[i].items():
                total[g] = total.get(g, 0) + f
        to_render += sorted(total.items(), key=lambda x: -x[1])[:args.render]
    images = render_notation(to_render, key_by_freq=True)

    fig = show_all(tunes, sel, heads, outs, H, O, ranks, C, pos, index, images, args.grid)
    hist_fig = show_similarity_histograms(tunes, H, O)
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save / 'head_vs_output.png', dpi=110)
        hist_fig.savefig(args.save / 'similarity_histograms.png', dpi=110)
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
