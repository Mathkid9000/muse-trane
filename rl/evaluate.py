"""Compare models on the analysis metric itself (similarity_analysis.calculate_frequencies, centred over
the generated set exactly as analysis/model_similarity.py does) plus the RL reward's view.

    python rl/evaluate.py                                  # bird-v1 vs bird-v1-rlvr on held-out heads
    python rl/evaluate.py --models bird-v1 --split all --n 8
    python rl/evaluate.py --weights model_best.pt          # load <checkpoint>/model_best.pt for RL models
"""
import common
import argparse
import pickle
import numpy as np
import torch

import config
from tokenizer import QuintupleTokenizer, tokens_to_score
from tune import Tune
from model_util import load_model_only, model_paths
from similarity_analysis import calculate_frequencies, summarize
from rollout import Constraints, sample
from reward import Reference


def decode_tune(ids, task, tokenizer):
    try:
        _, score = tokens_to_score(tokenizer.decode(np.asarray(ids)), export=False)
    except Exception:
        return None
    tune = Tune(task.file)
    tune.load_from_score(score, head_length=task.head_length)
    return tune


def generate(model, cons, tokenizer, tasks, args):
    tunes = []
    for task in tasks:
        seqs = sample(model, cons, task.seed, args.n, max_rows=len(task.seed) + config.SOLO_ROWS,
                      min_rows=len(task.seed) + config.MIN_SOLO_ROWS,
                      temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                      stride=config.SCORE_STRIDE)
        for s in seqs:
            t = decode_tune(s, task, tokenizer)
            if t is not None: tunes.append(t)
        print(f'  {task.file}: {len(seqs)} rollouts')
    return tunes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=[ config.BASE_MODEL, config.RL_MODEL ])
    ap.add_argument('--split', choices=('heldout', 'train', 'all'), default='heldout')
    ap.add_argument('--n', type=int, default=8, help='generations per head')
    ap.add_argument('--temperature', type=float, default=0.8)   # model_similarity's sampling settings
    ap.add_argument('--top-k', type=int, default=30)
    ap.add_argument('--top-p', type=float, default=0.9)
    ap.add_argument('--weights', default=None, help='weights file inside the checkpoint dir, eg. model_best.pt')
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = QuintupleTokenizer()
    cons = Constraints(tokenizer, device)
    train, heldout = common.load_split(common.load_tasks(tokenizer))
    tasks = { 'heldout': heldout, 'train': train, 'all': train + heldout }[args.split]
    ref = Reference.load()
    (common.RL_DIR / 'logs').mkdir(exist_ok=True)

    for name in args.models:
        cache = common.RL_DIR / 'logs' / f'gen_{name}_{args.weights or "model"}_{args.split}_{args.n}.pkl'
        if cache.exists() and not args.no_cache:
            tunes = pickle.loads(cache.read_bytes())
        else:
            model, _ = load_model_only(name, device)
            if args.weights:
                model.load_state_dict(torch.load(model_paths(name)['dir'] / args.weights, map_location=device, weights_only=True))
            print(f'Generating with {name} ({args.split}, {len(tasks)} heads x {args.n})')
            tunes = generate(model, cons, tokenizer, tasks, args)
            cache.write_bytes(pickle.dumps(tunes))

        print(f'\n===== {name} =====')
        notes = [ len(t.output_notes) for t in tunes ]
        print(f'{len(tunes)} tunes, solo notes median {np.median(notes):.0f} (min {min(notes)}, max {max(notes)})')
        print('-- analysis metric (centred over this generated set) --')
        direct, crossed = calculate_frequencies(tunes)
        summarize(direct, crossed)
        print('-- reward view (centred by the Omnibook, vs Omnibook heads) --')
        infos = [ ref.score(t.output_notes, t.title)[1] for t in tunes ]
        for k in sorted(infos[0]):
            print(f'  {k:<22} {np.mean([ i[k] for i in infos ]):8.4f}')


if __name__ == '__main__':
    main()
