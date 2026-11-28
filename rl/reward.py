"""Verifiable rewards for head <-> solo similarity.

Reference reproduces the exact analysis metric (used by --check and evaluate.py); DenseReference (below)
is the training reward, a denser, offset-free contrast built from the same kernels -- see its docstring.
The rest of this docstring describes the exact metric.

The analysis centres every frequency vector by a leave-one-out corpus mean and correlates through
F = S ** POWER over the corpus vocabulary V. With dh = A H (A = I - W, W the leave-one-out weights)
and a new solo o centred by the Omnibook output mean W O, everything splits into terms that are
either precomputed once from the Omnibook or need only S(solo grams, V) and S(solo grams, solo grams):

    <dh_k, F do_i> = (A kH(o))_k - (A C_HO W^T)_ki        kH(o) = sum_u o_u S(u, V)^P H^T
    <do_i, F do_i> = o S_UU o - 2 (W kO(o))_i + (W C_OO W^T)_ii
    <dh_k, F dh_k> = (A C_HH A^T)_kk

where C = [H; O] F [H; O]^T comes from similarity_analysis.kernel_grams. `python rl/reward.py --check`
recomputes the analysis numbers for Omnibook solos through this path to show the two agree.
"""
import common  # noqa: F401  (paths)
import pickle
import time
import numpy as np

import kernels
import config
from tune import Tune, Note
from similarity_analysis import N, POWER, TYPES, cached_tunes, base_title, frequency_matrix, kernel_grams

REFERENCE_CACHE = common.RL_DIR / 'reference_cache.pkl'


def _pad(E, L):
    if E.shape[1] == L: return E
    out = np.zeros((E.shape[0], L, 3), np.float32)
    out[:, :E.shape[1]] = E
    return out


def _encode(grams):
    """(contour E, lengths), (rhythm E, lengths) for gram ids."""
    return Note.encode_grams(grams, Note.contour_elements), Note.encode_grams(grams, Note.rhythm_elements)


def _sim_rows(E_rows, m_rows, E_cols, m_cols, w):
    """S(rows, cols) via kernels.similarity_block on the stacked encoding (rows first)."""
    L = max(E_rows.shape[1], E_cols.shape[1])
    E = np.concatenate([ _pad(E_rows, L), _pad(E_cols, L) ])
    m = np.concatenate([ m_rows, m_cols ])
    return kernels.similarity_block(E, m, w, 0, len(E_rows))[:, len(E_rows):]


class Reference:
    """Everything about the Omnibook the reward needs, built once and cached to disk."""

    def __init__(self, tunes):
        self.titles = [ t.title for t in tunes ]
        self.T = T = len(tunes)
        self.vocab = sorted(set(Tune.count_head_n_grams(tunes, N)) | set(Tune.count_output_n_grams(tunes, N)))
        index = { g: i for i, g in enumerate(self.vocab) }
        bases = np.array([ base_title(t) for t in self.titles ])
        self.same = bases[:, None] == bases[None, :]
        self.H = frequency_matrix(tunes, lambda t: t.head_notes, index).astype(np.float64)
        self.O = frequency_matrix(tunes, lambda t: t.output_notes, index).astype(np.float64)
        # leave_one_out_mean(X, same) == W @ X
        self.W = (1 - self.same) / (T - self.same.sum(1, keepdims=True))
        self.A = np.eye(T) - self.W
        self.head_grams = [ set(Tune.count_n_grams(t.head_notes, N)) for t in tunes ]

        t0 = time.time()
        G = kernel_grams(self.vocab, np.vstack([ self.H, self.O ]))
        print(f'reference kernel {time.time() - t0:.1f}s')
        self.dHH, self.HMO, self.MOMO, self.parker_direct = {}, {}, {}, {}
        for name, g in G.items():
            C_HH, C_HO, C_OO = g[:T, :T], g[:T, T:], g[T:, T:]
            self.dHH[name] = np.diag(self.A @ C_HH @ self.A.T)          # <dh_k, F dh_k>
            self.HMO[name] = self.A @ C_HO @ self.W.T                     # [k, i] = <dh_k, F mO_i>
            self.MOMO[name] = np.diag(self.W @ C_OO @ self.W.T)          # <mO_i, F mO_i>
            dOO = np.diag(self.A @ C_OO @ self.A.T)
            with np.errstate(invalid='ignore', divide='ignore'):
                self.parker_direct[name] = np.diag(self.A @ C_HO @ self.A.T) / np.sqrt(self.dHH[name] * dOO)
        self._init_runtime()

    def _init_runtime(self):
        (self.Ec, self.mc), (self.Er, self.mr) = _encode(self.vocab)
        self.index = { t: i for i, t in enumerate(self.titles) }
        self._proj = {}   # gram id -> {type: (kH (T,), kO (T,))}
        # direct r is capped at this Parker quantile: past where Parker himself sits, more r means copying
        self.target = { name: config.TARGET_R.get(name) if config.TARGET_R else
                        float(np.nanquantile(self.parker_direct[name], config.TARGET_QUANTILE)) for name in TYPES }

    def __getstate__(self):
        return { k: v for k, v in self.__dict__.items() if k not in ('Ec', 'mc', 'Er', 'mr', '_proj', 'index') }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_runtime()

    @classmethod
    def load(cls):
        tunes = cached_tunes()
        key = (2, N, POWER, tuple(t.title for t in tunes))   # bump the leading version when the state changes
        # the state dict is pickled, not the object: the class lives in __main__ when this file is run directly
        if REFERENCE_CACHE.exists():
            with open(REFERENCE_CACHE, 'rb') as f: cached_key, state = pickle.load(f)
            if cached_key == key:
                ref = cls.__new__(cls)
                ref.__setstate__(state)
                return ref
        ref = cls(tunes)
        with open(REFERENCE_CACHE, 'wb') as f: pickle.dump((key, ref.__getstate__()), f)
        return ref

    # --- per-solo ---
    def _projections(self, grams):
        """Fill the cache with S(u, V)^P projected onto the Omnibook heads and outputs."""
        new = [ g for g in grams if g not in self._proj ]
        if not new: return
        (Ec, mc), (Er, mr) = _encode(new)
        c = _sim_rows(Ec, mc, self.Ec, self.mc, kernels.CONTOUR_W).astype(np.float64)
        r = _sim_rows(Er, mr, self.Er, self.mr, kernels.RHYTHM_W).astype(np.float64)
        proj = { name: (F @ self.H.T, F @ self.O.T) for name, F in ((n, comb(c, r) ** POWER) for n, comb in TYPES.items()) }
        for j, g in enumerate(new):
            self._proj[g] = { name: (kH[j], kO[j]) for name, (kH, kO) in proj.items() }

    def correlations(self, notes, head):
        """{type: (T,) r between every Omnibook head k and this solo, centred as a solo for tune `head`}."""
        counts = Tune.count_n_grams(notes, N)
        if not counts: return None
        grams = list(counts)
        o = np.array([ counts[g] for g in grams ], np.float64)
        o /= o.sum()
        self._projections(grams)
        (Ec, mc), (Er, mr) = _encode(grams)
        c_uu = _sim_rows(Ec, mc, Ec, mc, kernels.CONTOUR_W).astype(np.float64)
        r_uu = _sim_rows(Er, mr, Er, mr, kernels.RHYTHM_W).astype(np.float64)
        out = {}
        for name, comb in TYPES.items():
            kH = np.stack([ self._proj[g][name][0] for g in grams ])   # (U, T)
            kO = np.stack([ self._proj[g][name][1] for g in grams ])
            num = self.A @ (o @ kH) - self.HMO[name][:, head]          # (T,) <dh_k, F do>
            dodo = o @ (comb(c_uu, r_uu) ** POWER) @ o - 2 * (self.W @ (o @ kO))[head] + self.MOMO[name][head]
            den = self.dHH[name] * dodo
            with np.errstate(invalid='ignore', divide='ignore'):
                out[name] = np.clip(np.where(den > 0, num / np.sqrt(np.abs(den)), np.nan), -1, 1)
        return out

    def score(self, notes, title):
        """Reward and its components for solo `notes` played over the head of Omnibook tune `title`."""
        head = self.index[title]
        info = { 'notes': len(notes) }
        R = 0.0
        corr = self.correlations(notes, head)
        others = ~self.same[head]
        for name, w in config.TYPE_WEIGHTS.items():
            if corr is None: d = c = 0.0
            else:
                d = float(np.nan_to_num(corr[name][head]))
                c = float(np.nanmean(corr[name][others])) if np.isfinite(corr[name][others]).any() else 0.0
            info[f'direct_{name}'], info[f'crossed_{name}'] = d, c
            R += w * (min(d, self.target[name]) - config.CROSSED_WEIGHT * c)
        counts = Tune.count_n_grams(notes, N)
        total = sum(counts.values())
        copy = sum(v for g, v in counts.items() if g in self.head_grams[head]) / total if total else 0.0
        info['copy'] = copy
        R -= config.COPY_WEIGHT * max(0.0, copy - config.COPY_FREE)
        R -= config.LENGTH_WEIGHT * max(0.0, 1 - len(notes) / config.MIN_SOLO_NOTES)
        info['reward'] = R
        return R, info


def _freq(tunes, notes_of, n, index):
    X = np.zeros((len(tunes), len(index)), np.float64)
    for i, t in enumerate(tunes):
        for g, f in Tune.count_n_grams(notes_of(t), n).items(): X[i, index[g]] = f
    return X / np.maximum(X.sum(1, keepdims=True), 1)


def _gram_kernel(vocab, D, power):
    """{type: D F D^T} with F = S ** power, streamed like similarity_analysis.kernel_grams (which fixes POWER)."""
    contour = Note.encode_grams(vocab, Note.contour_elements)
    rhythm = Note.encode_grams(vocab, Note.rhythm_elements)
    G = { name: np.zeros((len(D), len(D))) for name in TYPES }
    for i0 in range(0, len(vocab), 1024):
        i1 = min(i0 + 1024, len(vocab))
        c = kernels.similarity_block(*contour, kernels.CONTOUR_W, i0, i1)
        r = kernels.similarity_block(*rhythm, kernels.RHYTHM_W, i0, i1)
        for name, comb in TYPES.items():
            G[name] += D[:, i0:i1] @ (comb(c, r).astype(np.float64) ** power @ D.T)
    return G


class DenseReference:
    """Training reward: how much more a solo resembles ITS head than the other Omnibook heads.

    The exact metric (Reference) is a poor training signal: for bird-v1's solos its solo-dependent part is
    ~0.1% of the numerator, the rest a fixed per-head offset from centring by Parker's mean solo, which GRPO
    turns into noise. So, for training:

    1. dense kernel -- shorter grams (config.REWARD_N) and a softer kernel (config.REWARD_POWER) give partial
       credit to near-quotes. 4-grams with S^3 separate Parker's solos from other tunes' as well as the exact
       metric does (Combination: positive for 70% of solos, t = 2.9 vs 2.6 for 6-grams S^10).
    2. no offset -- no corpus-mean centring. With cos_k(o) = <h_k, F o> / sqrt(<h_k F h_k><o F o>), the raw
       kernel cosine between solo o and Omnibook head k,
            R(o, i) = cos_i(o) - mean_{k not a version of i} cos_k(o)
       is how much more the solo resembles its own head than heads in general.
    `python rl/reward.py --dense` prints Parker's distribution of R.
    """

    def __init__(self, tunes, n, power):
        self.n, self.power = n, power
        self.titles = [ t.title for t in tunes ]
        T = len(tunes)
        self.vocab = sorted(set(Tune.count_head_n_grams(tunes, n)) | set(Tune.count_output_n_grams(tunes, n)))
        index = { g: i for i, g in enumerate(self.vocab) }
        bases = np.array([ base_title(t) for t in self.titles ])
        self.same = bases[:, None] == bases[None, :]
        self.H = _freq(tunes, lambda t: t.head_notes, n, index)
        O = _freq(tunes, lambda t: t.output_notes, n, index)
        self.head_grams = [ set(Tune.count_n_grams(t.head_notes, N)) for t in tunes ]   # verbatim copy check, N = 6
        t0 = time.time()
        G = _gram_kernel(self.vocab, np.vstack([ self.H, O ]), power)
        print(f'dense reference kernel ({n}-grams, S^{power}) {time.time() - t0:.1f}s')
        self.hFh, self.parker = {}, {}
        for name, g in G.items():
            hFh, oFo = np.diag(g[:T, :T]), np.diag(g[T:, T:])
            cos = g[:T, T:] / np.sqrt(np.outer(hFh, oFo))          # [k, j]: head k vs Parker solo j
            self.hFh[name] = hFh
            self.parker[name] = np.array([ self._contrast(cos[:, j], j) for j in range(T) ])
        self._init_runtime()

    def _contrast(self, cos, head):
        return float(cos[head] - cos[~self.same[head]].mean())

    def _init_runtime(self):
        (self.Ec, self.mc), (self.Er, self.mr) = _encode(self.vocab)
        self.index = { t: i for i, t in enumerate(self.titles) }
        self._proj = {}
        # each contrast is capped at Parker's quantile: past where Parker himself sits, more is copying
        self.target = { name: float(np.quantile(self.parker[name], config.TARGET_QUANTILE)) for name in TYPES }
        # spread of the full reward over Parker's solos: the scale real head-awareness differences live on
        self.parker_reward = sum(w * np.minimum(self.parker[name], self.target[name]) for name, w in config.TYPE_WEIGHTS.items())
        self.adv_floor = config.ADV_STD_FLOOR * float(self.parker_reward.std())

    __getstate__, __setstate__ = Reference.__getstate__, Reference.__setstate__

    @classmethod
    def load(cls, n=None, power=None):
        n, power = n or config.REWARD_N, power or config.REWARD_POWER
        tunes = cached_tunes()
        key = ('dense', 2, n, power, tuple(t.title for t in tunes))
        path = common.RL_DIR / f'dense_reference_n{n}_p{power}.pkl'
        if path.exists():
            with open(path, 'rb') as f: cached_key, state = pickle.load(f)
            if cached_key == key:
                ref = cls.__new__(cls)
                ref.__setstate__(state)
                return ref
        ref = cls(tunes, n, power)
        with open(path, 'wb') as f: pickle.dump((key, ref.__getstate__()), f)
        return ref

    def _projections(self, grams):
        new = [ g for g in grams if g not in self._proj ]
        if not new: return
        (Ec, mc), (Er, mr) = _encode(new)
        c = _sim_rows(Ec, mc, self.Ec, self.mc, kernels.CONTOUR_W).astype(np.float64)
        r = _sim_rows(Er, mr, self.Er, self.mr, kernels.RHYTHM_W).astype(np.float64)
        proj = { name: (comb(c, r) ** self.power) @ self.H.T for name, comb in TYPES.items() }
        for j, g in enumerate(new):
            self._proj[g] = { name: p[j] for name, p in proj.items() }

    def cosines(self, notes):
        """{type: (T,) kernel cosine between this solo and every Omnibook head}, or None for no grams."""
        counts = Tune.count_n_grams(notes, self.n)
        if not counts: return None
        grams = list(counts)
        o = np.array([ counts[g] for g in grams ], np.float64)
        o /= o.sum()
        self._projections(grams)
        (Ec, mc), (Er, mr) = _encode(grams)
        c_uu = _sim_rows(Ec, mc, Ec, mc, kernels.CONTOUR_W).astype(np.float64)
        r_uu = _sim_rows(Er, mr, Er, mr, kernels.RHYTHM_W).astype(np.float64)
        out = {}
        for name, comb in TYPES.items():
            hFo = o @ np.stack([ self._proj[g][name] for g in grams ])
            oFo = o @ (comb(c_uu, r_uu) ** self.power) @ o
            out[name] = hFo / np.sqrt(self.hFh[name] * oFo)
        return out

    def score(self, notes, title):
        head = self.index[title]
        info = { 'notes': len(notes) }
        cos = self.cosines(notes)
        R = 0.0
        for name, w in config.TYPE_WEIGHTS.items():
            s = 0.0 if cos is None else self._contrast(cos[name], head)
            info[f'contrast_{name}'] = s
            R += w * min(s, self.target[name])
        counts = Tune.count_n_grams(notes, N)
        total = sum(counts.values())
        copy = sum(v for g, v in counts.items() if g in self.head_grams[head]) / total if total else 0.0
        info['copy'] = copy
        R -= config.COPY_WEIGHT * max(0.0, copy - config.COPY_FREE)
        R -= config.LENGTH_WEIGHT * max(0.0, 1 - len(notes) / config.MIN_SOLO_NOTES)
        info['reward'] = R
        return R, info


# --- decoding rollouts (top-level so a process pool can pickle it) ---
def decode_solo(args):
    """(ids (n,5), head_length) -> solo Notes, exactly as model_similarity builds a generated Tune; None on failure."""
    ids, head_length = args
    try:
        from tokenizer import tokens_to_score
        global _TOKENIZER
        if '_TOKENIZER' not in globals():
            from tokenizer import QuintupleTokenizer
            _TOKENIZER = QuintupleTokenizer()
        _, score = tokens_to_score(_TOKENIZER.decode(ids), export=False)
        tune = Tune('rollout')
        tune.load_from_score(score, head_length=head_length)
        return tune.output_notes
    except Exception:
        return None


class Rewarder:
    def __init__(self, workers=config.REWARD_WORKERS):
        self.ref = DenseReference.load()
        self.pool = None
        if workers > 0:
            import multiprocessing as mp
            self.pool = mp.get_context('spawn').Pool(workers)

    def __call__(self, task, seqs):
        jobs = [ (np.asarray(s), task.head_length) for s in seqs ]
        solos = self.pool.map(decode_solo, jobs) if self.pool else [ decode_solo(j) for j in jobs ]
        out = []
        for notes in solos:
            if notes is None: out.append((config.FAIL_REWARD, { 'reward': config.FAIL_REWARD, 'failed': 1 }))
            else: out.append(self.ref.score(notes, task.file))
        return out


if __name__ == '__main__':
    import argparse
    from similarity_analysis import calculate_frequencies
    ap = argparse.ArgumentParser(description='Check the per-solo reward reproduces similarity_analysis on Omnibook solos.')
    ap.add_argument('--check', type=int, default=6, help='number of Omnibook solos to recompute')
    ap.add_argument('--dense', action='store_true', help="report the training reward on Parker's solos instead")
    args = ap.parse_args()

    if args.dense:
        ref = DenseReference.load()
        print(f'training reward: {ref.n}-grams, S^{ref.power}; contrast = cos(own head) - mean cos(other heads)')
        for n in TYPES:
            v = ref.parker[n]
            print(f'  Parker {n:<12} mean {v.mean():+.4f}  sd {v.std():.4f}  >0 in {np.mean(v > 0):.0%}  '
                  f't {v.mean() / v.std() * np.sqrt(len(v)):.1f}  cap {ref.target[n]:.4f}')
        print(f'  Parker full reward sd {ref.parker_reward.std():.4f} -> advantage std floor {ref.adv_floor:.4f}')
        raise SystemExit

    ref = Reference.load()
    tunes = cached_tunes()
    direct, crossed = calculate_frequencies(tunes)
    print(f'\n{"tune":<28}' + ''.join(f'{n[:5]:>9}{"(ref)":>8}' for n in TYPES) + f'{"copy":>7}{"reward":>8}')
    worst = 0.0
    for i, t in enumerate(tunes[:args.check]):
        t0 = time.time()
        corr = ref.correlations(t.output_notes, i)
        R, info = ref.score(t.output_notes, t.title)
        row = f'{t.title[:27]:<28}'
        for n in TYPES:
            row += f'{corr[n][i]:9.4f}{direct[n][i]:8.4f}'
            if np.isfinite(direct[n][i]): worst = max(worst, abs(corr[n][i] - direct[n][i]))
        print(row + f'{info["copy"]:7.2f}{R:8.3f}   ({time.time() - t0:.1f}s)')
    print(f'max |reward path - analysis| = {worst:.2e}')
    for n in TYPES:
        print(f'Parker direct {n:<12} mean {np.nanmean(ref.parker_direct[n]):.3f}  '
              f'p{int(config.TARGET_QUANTILE * 100)} {np.nanquantile(ref.parker_direct[n], config.TARGET_QUANTILE):.3f}  cap {ref.target[n]:.3f}')
    copies = [ ref.score(t.output_notes, t.title)[1]['copy'] for t in tunes ]
    print(f'Parker copy fraction (solo 6-grams found verbatim in head): mean {np.mean(copies):.3f}, max {np.max(copies):.3f}')
