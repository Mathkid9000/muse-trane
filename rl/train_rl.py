"""GRPO-style RLVR: sample a group of solos per Omnibook head, reward head <-> solo kernel correlation
(rl/reward.py), and push up the log-prob of above-group-average solos, with a KL penalty to the frozen
base model.

    python rl/train_rl.py                      # full run with rl/config.py
    python rl/train_rl.py --steps 1 --heads 1 --group 2 --solo-rows 40 --min-solo-rows 20   # smoke test
    python rl/train_rl.py --resume             # continue from the saved RL checkpoint
"""
import common
import argparse
import copy
import json
import time
import numpy as np
import torch

import config
from helpers import SEQ_LEN
from tokenizer import QuintupleTokenizer
from model_util import load_model_only, model_paths, save_model
from rollout import Constraints, sample, build_windows, row_logprobs
from reward import Rewarder

LOG_DIR = common.RL_DIR / 'logs'


def rollout_group(policy, cons, task, seed, n, rewarder, args):
    seqs = sample(policy, cons, seed, n, max_rows=len(seed) + args.solo_rows,
                  min_rows=len(seed) + args.min_solo_rows, temperature=config.TEMPERATURE, stride=config.SCORE_STRIDE)
    return seqs, rewarder(task, seqs)


def group_advantages(rewards):
    R = np.asarray(rewards, np.float64)
    std = R.std()
    return (R - R.mean()) / std if std > 1e-6 else np.zeros_like(R)


def policy_step(policy, ref, cons, batch, opt, args, device):
    """One on-policy update. batch: list of (ids, solo_start, min_rows, advantage)."""
    policy.eval()   # dropout off: score the same distribution that was sampled; gradients still flow
    x, y, keep, blocked, owner = build_windows([ (b[0], b[1]) for b in batch ], cons, [ b[2] for b in batch ],
                                               SEQ_LEN, config.SCORE_STRIDE)
    adv = torch.tensor([ b[3] for b in batch ], dtype=torch.float32)
    norm = args.solo_rows * len(batch)          # Dr.GRPO-style constant normalizer: no length bias
    opt.zero_grad(set_to_none=True)
    kl_sum, lp_sum, rows = 0.0, 0.0, int(keep.sum())
    for i in range(0, len(x), config.MICRO_BATCH):
        sl = slice(i, i + config.MICRO_BATCH)
        xb, yb, kb, bb = (t[sl].to(device) for t in (x, y, keep, blocked))
        ab = adv[owner[sl]].to(device)[:, None]
        with torch.no_grad():
            ref_lp = row_logprobs(ref, cons, xb, yb, kb, bb, config.TEMPERATURE)
        lp = row_logprobs(policy, cons, xb, yb, kb, bb, config.TEMPERATURE)
        d = (ref_lp - lp).clamp(max=20)
        k3 = torch.where(kb, torch.exp(d) - d - 1, torch.zeros_like(d))   # unbiased, >= 0 KL estimate
        loss = (-(ab * lp) + config.KL_BETA * k3)[kb].sum() / norm
        loss.backward()
        kl_sum += float(k3.detach().sum())
        lp_sum += float(lp.detach()[kb].sum())
    grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), config.CLIP_NORM))
    opt.step()
    return { 'kl_per_row': kl_sum / max(rows, 1), 'logp_per_row': lp_sum / max(rows, 1),
             'grad_norm': grad_norm, 'windows': len(x), 'scored_rows': rows }


@torch.no_grad()
def evaluate(policy, cons, tasks, rewarder, args):
    infos = []
    for task in tasks:
        _, scored = rollout_group(policy, cons, task, task.seed, config.EVAL_GROUP_SIZE, rewarder, args)
        infos += [ s[1] for s in scored ]
    return mean_info(infos, 'eval_')


def mean_info(infos, prefix=''):
    keys = sorted({ k for i in infos for k in i })
    return { prefix + k: float(np.mean([ i.get(k, 0.0) for i in infos ])) for k in keys }


def save(policy, cfg, opt, heldout, state):
    paths = model_paths(config.RL_MODEL)
    paths['dir'].mkdir(parents=True, exist_ok=True)
    save_model(config.RL_MODEL, cfg, policy, opt, None, sorted(t.name for t in heldout))
    (paths['dir'] / 'rl_state.json').write_text(json.dumps(state, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=config.MAX_STEPS)
    ap.add_argument('--heads', type=int, default=config.HEADS_PER_STEP)
    ap.add_argument('--group', type=int, default=config.GROUP_SIZE)
    ap.add_argument('--solo-rows', type=int, default=config.SOLO_ROWS)
    ap.add_argument('--min-solo-rows', type=int, default=config.MIN_SOLO_ROWS)
    ap.add_argument('--eval-every', type=int, default=config.EVAL_EVERY)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = QuintupleTokenizer()
    cons = Constraints(tokenizer, device)
    train, heldout = common.load_split(common.load_tasks(tokenizer))
    print(f'{len(train)} train heads, {len(heldout)} held out: {[ t.file for t in heldout ]}')
    transposer = common.Transposer()
    rewarder = Rewarder()

    ref, cfg = load_model_only(config.BASE_MODEL, device)
    ref.eval().requires_grad_(False)
    policy = copy.deepcopy(ref).requires_grad_(True)
    opt = torch.optim.AdamW(policy.parameters(), lr=config.LR, weight_decay=0.0)
    state = { 'step': 0, 'best_eval_reward': -float('inf') }
    paths = model_paths(config.RL_MODEL)
    if args.resume and paths['weights'].exists():
        policy.load_state_dict(torch.load(paths['weights'], map_location=device, weights_only=True))
        if paths['optimizer'].exists():
            opt.load_state_dict(torch.load(paths['optimizer'], map_location=device, weights_only=True))
        if (paths['dir'] / 'rl_state.json').exists():
            state = json.loads((paths['dir'] / 'rl_state.json').read_text())
        print(f'Resumed {config.RL_MODEL} at step {state["step"]}')

    LOG_DIR.mkdir(exist_ok=True)
    log = open(LOG_DIR / 'train_log.jsonl', 'a')
    while state['step'] < args.steps:
        t0 = time.time()
        batch, infos = [], []
        for idx in rng.choice(len(train), size=min(args.heads, len(train)), replace=False):
            task = train[idx]
            seed = transposer.random(task.seed, rng) if config.TRANSPOSE_SEEDS else task.seed
            seqs, scored = rollout_group(policy, cons, task, seed, args.group, rewarder, args)
            A = group_advantages([ s[0] for s in scored ])
            batch += [ (seqs[g], len(seed), len(seed) + args.min_solo_rows, float(A[g])) for g in range(len(seqs)) ]
            infos += [ s[1] for s in scored ]
        t_roll = time.time() - t0
        stats = policy_step(policy, ref, cons, batch, opt, args, device)
        state['step'] += 1
        entry = { 'step': state['step'], **mean_info(infos), **stats,
                  'rollout_s': round(t_roll, 1), 'step_s': round(time.time() - t0, 1) }

        if state['step'] % args.eval_every == 0 or state['step'] == args.steps:
            entry.update(evaluate(policy, cons, heldout, rewarder, args))
            save(policy, cfg, opt, heldout, state)       # latest -> model.pt (what evaluate.py loads)
            if entry['eval_reward'] > state['best_eval_reward']:
                state['best_eval_reward'] = entry['eval_reward']
                state['best_step'] = state['step']
                torch.save(policy.state_dict(), paths['dir'] / 'model_best.pt')
                (paths['dir'] / 'rl_state.json').write_text(json.dumps(state, indent=2))
        log.write(json.dumps(entry) + '\n'); log.flush()
        print(' | '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in entry.items()
                         if k in ('step', 'reward', 'direct_Combination', 'crossed_Combination', 'direct_Rhythmic',
                                  'copy', 'notes', 'kl_per_row', 'grad_norm', 'step_s', 'eval_reward',
                                  'eval_direct_Combination', 'eval_crossed_Combination')))


if __name__ == '__main__':
    main()
