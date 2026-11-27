"""Batched rollouts and their log-probabilities.

`sample` is generate.generate_score for a batch of rollouts sharing one seed, under the same
structural masks, with a KV cache. Past SEQ_LEN rows its context window slides in blocks (window_start)
rather than every row, so a teacher-forced window can reproduce each row's context exactly. `Constraints.mask` is shared by sampling and scoring, so the log-prob the
policy gradient uses is the log-prob of the masked distribution the row was actually drawn
from (forced fields have log-prob exactly 0 and drop out on their own)."""
import common  # noqa: F401  (paths)
import torch
import torch.nn.functional as F
from helpers import SEQ_LEN
from model import apply_rotary_pos_emb


class Constraints:
    def __init__(self, tokenizer, device):
        V = tokenizer.field_sizes
        self.empty = tokenizer.pad_ids
        self.eos = tokenizer.vocabs[0]['EOS']
        self.pad_row = torch.tensor(self.empty, dtype=torch.long, device=device)

        def flags(size, ids):
            t = torch.zeros(size, dtype=torch.bool, device=device)
            t[sorted(ids)] = True
            return t
        self.struct = flags(V[0], tokenizer.struct_f0)
        self.instr = flags(V[0], tokenizer.instr_f0)
        self.voice = flags(V[4], tokenizer.voice_f4)
        self.allow = {}
        for f, table in ((3, tokenizer.event_f3), (4, tokenizer.event_f4)):
            a = torch.zeros(V[2], V[f], dtype=torch.bool, device=device)
            for f2, ids in table.items(): a[f2, ids] = True
            a[~a.any(1)] = True     # unknown field-2 type: leave unconstrained rather than all -inf
            self.allow[f] = a
        self.sizes = V

    def mask(self, i, prev, eos_blocked):
        """Allowed ids for field i, shape (..., V_i), given the row's earlier fields `prev` (list of (...) long)
        and `eos_blocked` (...) bool. Mirrors the masking in generate.generate_score."""
        V = self.sizes[i]
        if i == 0:
            shape = eos_blocked.shape
            m = torch.ones(*shape, V, dtype=torch.bool, device=self.pad_row.device)
            m[..., self.empty[0]] = False
            m[..., self.eos] &= ~eos_blocked
            return m
        struct, instr = self.struct[prev[0]], self.instr[prev[0]]
        only_empty = torch.zeros(V, dtype=torch.bool, device=struct.device)
        only_empty[self.empty[i]] = True
        if i in (1, 2):
            event = ~only_empty
            return torch.where(struct[..., None], only_empty, event)
        event = self.allow[i][prev[2]]
        if i == 3:
            return torch.where(struct[..., None], only_empty, event)
        structural = torch.where(instr[..., None], self.voice, only_empty)
        return torch.where(struct[..., None], structural, event)


def _draw(logits, temperature, top_k, top_p):
    if temperature <= 0: return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float('-inf'))
    probs = F.softmax(logits, dim=-1)
    if top_p and top_p < 1.0:
        sp, order = torch.sort(probs, descending=True, dim=-1)
        keep = (sp.cumsum(-1) - sp) < top_p          # == generate.sample_field's cut
        probs = torch.zeros_like(probs).scatter(-1, order, sp * keep)
    return torch.multinomial(probs, 1).squeeze(-1)


def window_start(t, seq_len=SEQ_LEN, stride=64):
    """First context row used to generate row t. Up to seq_len rows the context is everything; past that
    the window start advances in blocks of `stride` rows (context between seq_len - stride + 1 and seq_len
    rows), so a block can be decoded with one KV cache and scored by one teacher-forced window, exactly."""
    if t <= seq_len: return 0
    return ((t - seq_len - 1) // stride + 1) * stride


def windows(n_rows, start, seq_len=SEQ_LEN, stride=64):
    """(s, first) pairs covering targets start..n_rows-1, the same blocks window_start uses: window s feeds
    rows s..s+seq_len-1 and predicts rows s+1..s+seq_len, of which rows >= first are kept."""
    out = []
    if start <= seq_len: out.append((0, start))         # rows 1..seq_len, full context
    k = max(0, (max(start, seq_len + 1) - seq_len - 1) // stride)
    a = seq_len + 1 + k * stride
    while a < n_rows:
        s = (k + 1) * stride                             # == window_start(t) for t in [a, a + stride)
        out.append((s, max(a, start)))
        a += stride
        k += 1
    return out


def _trunk(model, ids, cache, pos):
    """model.hidden for new rows `ids` (B, t, 5) at window positions pos..pos+t-1, attending to `cache`
    (the window's earlier rows, per layer (k, v)) -- MusicTransformer's layers, unrolled with a KV cache."""
    embeds = [ emb(ids[..., i]) for i, emb in enumerate(model.field_embeddings) ]
    x = model.input_proj(torch.cat(embeds, dim=-1))
    t = x.shape[1]
    cos = model.rope.cos[pos:pos + t].view(1, 1, t, -1)
    sin = model.rope.sin[pos:pos + t].view(1, 1, t, -1)
    new_cache = []
    for li, layer in enumerate(model.decoders):
        attn = layer.self_attn
        out = layer.norm1(x)
        B = out.shape[0]
        q, k, v = (proj(out).view(B, t, -1, attn.head_dim).transpose(1, 2) for proj in (attn.q_proj, attn.k_proj, attn.v_proj))
        q, k = apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)
        if cache is not None:
            k = torch.cat([ cache[li][0], k ], dim=2)
            v = torch.cat([ cache[li][1], v ], dim=2)
        new_cache.append((k, v))
        if attn.num_groups > 1:
            k, v = k.repeat_interleave(attn.num_groups, dim=1), v.repeat_interleave(attn.num_groups, dim=1)
        # prefill: causal among the new rows; decode (t == 1): the new row sees the whole window
        o = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), is_causal=cache is None and t > 1)
        x = x + attn.out_proj(o.transpose(1, 2).reshape(B, t, -1))
        x = x + layer.mlp(layer.norm2(x))
    return model.norm(x), new_cache


@torch.no_grad()
def sample(model, cons, seed, n, max_rows, min_rows, temperature=1.0, top_k=0, top_p=1.0,
           seq_len=SEQ_LEN, stride=64):
    """n rollouts continuing `seed` ((s, 5) ids) until EOS or max_rows total rows; EOS is masked before
    min_rows. Returns a list of (len_i, 5) long tensors on CPU, each including the seed (and EOS if drawn).
    Context for row t is rows window_start(t)..t-1 (see there), decoded with a KV cache per window."""
    model.eval()
    device = cons.pad_row.device
    tokens = torch.as_tensor(seed, dtype=torch.long, device=device).unsqueeze(0).repeat(n, 1, 1)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    lengths = torch.full((n,), max_rows, dtype=torch.long)
    cache, cache_s, h = None, None, None
    while tokens.shape[1] < max_rows and not done.all():
        t = tokens.shape[1]
        s = window_start(t, seq_len, stride)
        if s != cache_s:                                       # new window: prefill rows s..t-1
            h, cache = _trunk(model, tokens[:, s:t], None, 0)
            cache_s = s
        blocked = torch.full((n,), t < min_rows, dtype=torch.bool, device=device)
        prev = []
        for i in range(model.num_fields):
            logits = model.field_logits(i, h[:, -1], prev).masked_fill(~cons.mask(i, prev, blocked), float('-inf'))
            prev.append(_draw(logits, temperature, top_k, top_p))
        row = torch.stack(prev, -1)
        row[done] = cons.pad_row
        tokens = torch.cat([ tokens, row.unsqueeze(1) ], dim=1)
        ended = (row[:, 0] == cons.eos) & ~done
        lengths[ended.cpu()] = tokens.shape[1]
        done |= ended
        if window_start(t + 1, seq_len, stride) == cache_s and t + 1 < max_rows:
            h, cache = _trunk(model, row.unsqueeze(1), cache, t - cache_s)
    return [ tokens[g, :min(int(lengths[g]), tokens.shape[1])].cpu() for g in range(n) ]


def build_windows(rollouts, cons, min_rows_of, seq_len=SEQ_LEN, stride=64):
    """Stack every scoring window of every rollout. rollouts: list of (ids (n,5), solo_start).
    Returns x, y (W, width, 5), keep (W, width) bool, eos_blocked (W, width) bool, owner (W,) long."""
    xs, ys, keeps, blocks, owner = [], [], [], [], []
    spans = [ (r, s, first) for r, (ids, start) in enumerate(rollouts) for s, first in windows(len(ids), start, seq_len, stride) ]
    # pad to the longest window actually present, not seq_len: with SEQ_LEN = 2048 a whole rollout is one
    # window, and padding ~700 rows out to 2048 would triple the scoring cost (padding is causal, so inert)
    width = max(min(seq_len, len(rollouts[r][0]) - 1 - s) for r, s, _ in spans)
    for r, s, first in spans:
        ids = rollouts[r][0]
        n = len(ids)
        y = ids[s + 1:s + 1 + seq_len]
        x = ids[s:s + len(y)]
        pad = cons.pad_row.cpu().expand(width - len(y), 5)
        target_idx = torch.arange(s + 1, s + 1 + width)
        xs.append(torch.cat([ x, pad ]))
        ys.append(torch.cat([ y, pad ]))
        keeps.append((target_idx >= first) & (target_idx < n))
        blocks.append(target_idx < min_rows_of[r])
        owner.append(r)
    return torch.stack(xs), torch.stack(ys), torch.stack(keeps), torch.stack(blocks), torch.tensor(owner)


def row_logprobs(model, cons, x, y, keep, eos_blocked, temperature=1.0):
    """(B, T) log-prob of each target row y under the masked, tempered chain, teacher-forced on x;
    0 where not `keep` (padding rows are -inf under the mask, and -inf * 0 would be nan)."""
    h = model.hidden(x)
    total = 0.0
    prev = [ y[..., j] for j in range(model.num_fields) ]
    for i in range(model.num_fields):
        logits = model.field_logits(i, h, prev[:i]).float() / temperature
        logits = logits.masked_fill(~cons.mask(i, prev[:i], eos_blocked), float('-inf'))
        lp = F.log_softmax(logits, -1).gather(-1, y[..., i:i + 1]).squeeze(-1)
        total = total + torch.where(keep, lp, torch.zeros_like(lp))
    return total
