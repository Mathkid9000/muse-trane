from melodic_analysis import NGramNote, Tune, load_tunes
import re
import numpy as np
from numba import njit, prange

def base_title(title):
    """'Au_Private_2.xml' -> 'Au_Private', so versions of the same tune group together."""
    return re.sub(r'(_\d+)?\.xml$', '', title)

@njit(cache=True)
def _pitched_nb(p, rest, out):
    m = 0
    for x in p:
        if x != rest:
            out[m] = x
            m += 1
    return m

@njit(cache=True)
def _sub_nb(x, y, intervals):
    if x == y:
        return 0.0
    if intervals:
        contour = 0.0 if np.sign(x) == np.sign(y) else 0.4
        return 0.3 + 0.3 * min(abs(x - y), 6) / 6 + contour
    if (x - y) % 12 == 0:
        return 0.2
    return 0.5 + 0.5 * min(abs(x - y), 6) / 6

@njit(cache=True)
def _align_nb(a, m, b, k, intervals, prev, cur):
    if m == 0 and k == 0:
        return 0.0
    for j in range(k + 1):
        prev[j] = j
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, k + 1):
            cur[j] = min(prev[j] + 1.0, cur[j-1] + 1.0, prev[j-1] + _sub_nb(a[i-1], b[j-1], intervals))
        prev, cur = cur, prev
    return min(1.0, prev[k] / max(m, k))

@njit(cache=True)
def _rhythm_nb(p1, d1, p2, d2, rest, prev, cur):
    m, k = len(p1), len(p2)
    prev[0] = 0.0
    for j in range(1, k + 1):
        prev[j] = prev[j-1] + min(1.0, 0.5 + d2[j-1])
    for i in range(1, m + 1):
        cur[0] = prev[0] + min(1.0, 0.5 + d1[i-1])
        for j in range(1, k + 1):
            cost = min(abs(np.log2(d1[i-1] / d2[j-1])) / 2, 1.0)
            if (p1[i-1] == rest) != (p2[j-1] == rest):
                cost += 0.5
            cur[j] = min(prev[j] + min(1.0, 0.5 + d1[i-1]), cur[j-1] + min(1.0, 0.5 + d2[j-1]),
                         prev[j-1] + min(cost, 1.0))
        prev, cur = cur, prev
    return 1.0 - min(1.0, prev[k] / max(m, k))

@njit(parallel=True, cache=True)
def _matrix_nb(HP, HD, OP, OD, rest, w_r, w_i, w_n, out):
    n = HP.shape[1]
    for h in prange(HP.shape[0]):
        prev, cur = np.empty(n + 1), np.empty(n + 1)
        a, b, q = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
        ia, ib = np.empty(n, np.int64), np.empty(n, np.int64)
        shifts = np.empty(n * n, np.int64)
        m1 = _pitched_nb(HP[h], rest, a)
        for i in range(m1 - 1):
            ia[i] = a[i+1] - a[i]
        for o in range(OP.shape[0]):
            m2 = _pitched_nb(OP[o], rest, b)
            for i in range(m2 - 1):
                ib[i] = b[i+1] - b[i]
            rhythm = _rhythm_nb(HP[h], HD[h], OP[o], OD[o], rest, prev, cur)
            interval = 1.0 - _align_nb(ia, max(m1 - 1, 0), ib, max(m2 - 1, 0), True, prev, cur)
            if m1 == 0 or m2 == 0:
                note = 1.0 if m1 == 0 and m2 == 0 else 0.0
            else:
                for i in range(m1):
                    for j in range(m2):
                        shifts[i * m2 + j] = a[i] - b[j]
                s = np.sort(shifts[:m1 * m2])
                best = np.inf
                for t in range(len(s)):
                    if t > 0 and s[t] == s[t-1]:
                        continue
                    for j in range(m2):
                        q[j] = b[j] + s[t]
                    best = min(best, _align_nb(a, m1, q, m2, False, prev, cur))
                    if best == 0.0:
                        break
                note = 1.0 - best
            score = (w_r * rhythm + w_i * interval + w_n * note) / (w_r + w_i + w_n)
            out[h, o] = max(0.0, min(1.0, score))

def _encode(gram_ids):
    notes = [NGramNote.gram_id_to_notes(g) for g in gram_ids]
    return (np.array([[x.pitch for x in ns] for ns in notes], dtype=np.int64),
            np.array([[float(x.dur) for x in ns] for ns in notes], dtype=np.float64))

def build_matrix(tunes, n):
    head_grams = list(Tune.count_head_n_grams(tunes, n))
    output_grams = list(Tune.count_output_n_grams(tunes, n))
    matrix = np.zeros((len(head_grams), len(output_grams)), dtype=np.float32)
    _matrix_nb(*_encode(head_grams), *_encode(output_grams), NGramNote.REST,
               NGramNote.SIM_W_RHYTHM, NGramNote.SIM_W_INTERVAL, NGramNote.SIM_W_NOTE, matrix)
    return { g:i for i,g in enumerate(head_grams) }, { g:i for i,g in enumerate(output_grams) }, np.floor(matrix)

def gram_vector(notes, gram_idxs, n):
    grams = Tune.count_n_grams(notes, n)
    vector = np.zeros(len(gram_idxs), dtype=np.float32)
    for gram, freq in grams.items():
        vector[gram_idxs[gram]] = freq
    return vector / max(vector.sum(), 1)

def main():
    n = 6
    tunes = load_tunes()
    head_gram_idxs, output_gram_idxs, similarity_matrix = build_matrix(tunes, n)

    heads = np.array([gram_vector(t.head_notes, head_gram_idxs, n) for t in tunes])
    outputs = np.array([gram_vector(t.output_notes, output_gram_idxs, n) for t in tunes])
    scores = heads @ similarity_matrix @ outputs.T  # [head tune, output tune]

    # direct head to output scores
    similarity_scores_same = [scores[i, i] for i in range(len(tunes))]

    # crossing outputs scores
    similarity_scores_crossed = []
    for i, head_tune in enumerate(tunes):
        avg_score = 0
        comparisons = 0
        for j, output_tune in enumerate(tunes):
            if base_title(head_tune.title) == base_title(output_tune.title):
                continue
            avg_score += scores[i, j]
            comparisons += 1
        similarity_scores_crossed.append(avg_score / comparisons)

    # display
    import matplotlib.pyplot as plt

    same = np.array(similarity_scores_same)
    crossed = np.array(similarity_scores_crossed)
    lo, hi = min(same.min(), crossed.min()), max(same.max(), crossed.max())
    pad = 0.05 * (hi - lo) or 0.05
    bins = np.linspace(lo - pad, hi + pad, 26)
    fig, (ax_s, ax_c) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, sharey=True,
                                     layout='constrained')
    fig.suptitle(f'Head vs. output n-gram similarity, all {len(tunes)} tunes', fontsize=11)
    for ax, vals, color, title in (
            (ax_s, same, '#2a78d6', 'Same tune: head vs. its own output'),
            (ax_c, crossed, '#e8743b', "Crossed: head vs. other tunes' outputs (averaged per tune)")):
        ax.hist(vals, bins=bins, color=color, edgecolor='white')
        ax.axvline(vals.mean(), color='#333', linestyle='--', linewidth=1)
        ax.text(vals.mean(), 0.97, f' mean {vals.mean():.3g}', transform=ax.get_xaxis_transform(),
                va='top', fontsize=9)
        ax.set_title(title, fontsize=10, loc='left')
        ax.set_ylabel('tunes')
        ax.spines[['top', 'right']].set_visible(False)
    ax_c.set_xlabel('similarity score')
    plt.show()

if __name__ == "__main__":
    main()