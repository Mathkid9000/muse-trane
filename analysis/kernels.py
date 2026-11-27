import numpy as np
from numba import njit, prange

# A gram is encoded as an (L, 3) float array of elements: (value, flag, indel_cost).
# Substituting two elements costs min(|value1 - value2| * w + 0.5 * (flag1 != flag2), 1).
CONTOUR_W = 1 / 12  # value = interval in semitones, flag = direction
RHYTHM_W = 1 / 2    # value = log2(duration), flag = is_rest


@njit(cache=True)
def edit_sim(e1, m1, e2, m2, w, prev, cur):
    """1 - normalized weighted edit distance between e1[:m1] and e2[:m2]; prev/cur are work rows of size >= m2 + 1."""
    if m1 == 0 and m2 == 0: return 1.0
    prev[0] = 0.0
    for j in range(m2):
        prev[j + 1] = prev[j] + e2[j, 2]
    for i in range(m1):
        cur[0] = prev[0] + e1[i, 2]
        for j in range(m2):
            sub = min(abs(e1[i, 0] - e2[j, 0]) * w + 0.5 * (e1[i, 1] != e2[j, 1]), 1.0)
            cur[j + 1] = min(prev[j] + sub, prev[j + 1] + e1[i, 2], cur[j] + e2[j, 2])
        prev, cur = cur, prev
    return max(0.0, 1.0 - prev[m2] / max(m1, m2))


@njit(cache=True)
def pair_sim(e1, e2, w):
    work = np.empty((2, len(e2) + 1))
    return edit_sim(e1, len(e1), e2, len(e2), w, work[0], work[1])


@njit(parallel=True, cache=True)
def similarity_block(E, m, w, i0, i1):
    """Similarities of grams i0..i1 against all grams: E is (V, L, 3) padded elements, m is (V,) lengths."""
    V, L = E.shape[0], E.shape[1]
    S = np.empty((i1 - i0, V), np.float32)
    for i in prange(i1 - i0):
        work = np.empty((2, L + 1))
        for j in range(V):
            S[i, j] = edit_sim(E[i0 + i], m[i0 + i], E[j], m[j], w, work[0], work[1])
    return S
