"""Hyperparameters for RLVR fine-tuning of the solo model toward head-aware solos."""

BASE_MODEL = 'bird-v1'          # policy init + frozen KL reference
RL_MODEL = 'bird-v1-rlvr'       # checkpoint name the RL policy is saved under

# --- rollouts ---
GROUP_SIZE = 8                  # solos sampled per head (GRPO group)
HEADS_PER_STEP = 4              # heads per optimizer step -> GROUP_SIZE * HEADS_PER_STEP rollouts
SOLO_ROWS = 400                 # rows generated after the head (Parker solos are ~300 notes)
MIN_SOLO_ROWS = 300             # EOS is masked until this many solo rows exist
TEMPERATURE = 1.0               # keep 1.0 for training: log-probs are scored at the same temperature
TRANSPOSE_SEEDS = True          # random key per rollout group; the reward is transposition invariant

# --- policy-gradient scoring ---
SCORE_STRIDE = 64               # past SEQ_LEN rows the context window advances in blocks of this many rows
                                # (sampling + scoring share it; larger = faster, shorter worst-case context)
MICRO_BATCH = 8                 # rollout windows per forward/backward (memory knob)
LR = 1e-5
KL_BETA = 0.05                  # k3 KL penalty to the frozen base model, per solo row
CLIP_NORM = 1.0
MAX_STEPS = 500

# --- reward (see reward.DenseReference) ---
REWARD_N = 4                    # gram length of the training reward (evaluate.py still uses the 6-gram metric)
REWARD_POWER = 3                # kernel F = S ** power; lower = more partial credit for near-quotes
TYPE_WEIGHTS = { 'Combination': 1.0, 'Rhythmic': 0.5 }
TARGET_QUANTILE = 0.75          # each contrast is capped at this quantile of Parker's own (beyond it is copying)
TARGET_R = None                 # exact-metric caps for reward.Reference (--check / evaluate.py); None -> quantile
COPY_FREE = 0.2                 # fraction of solo 6-grams allowed to appear verbatim in the head
COPY_WEIGHT = 2.0
MIN_SOLO_NOTES = 150            # linear penalty below this many solo notes
LENGTH_WEIGHT = 1.0
FAIL_REWARD = -1.0              # rollout could not be decoded into a score
ADV_STD_FLOOR = 0.25            # advantages divide by max(group std, this x Parker's reward sd): a group whose
                                # rewards barely differ gets small advantages instead of unit-size ones
REWARD_WORKERS = 0              # >0: decode rollouts to notes (music21) in a process pool

# --- split / eval ---
N_HELDOUT = 8                   # held-out tunes (bird-v1's own val tunes first), grouped by base title
EVAL_EVERY = 25
EVAL_GROUP_SIZE = 4
SEED = 0
