import music21
from collections import Counter
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

IMPORT_FOLDER = Path("./storage/data/omnibook/")

pitches = []
tempos = []
times = []
keys = []
lengths = []
voices = []
pitch_chords = []
chords = []
chord_kinds = []

# read all files
for file in sorted(IMPORT_FOLDER.iterdir()):
    print(f"parsing {file.name}")
    score = music21.converter.parse(file)
    flat = score.recurse()

    for n in flat.notesAndRests:
        pitches.extend([ p.midi for p in n.pitches ])
        pitch_chords.append(len(n.pitches))
        if len(n.pitches) > 1:
            print(n.offset, n.measureNumber, n.duration, [ p for p in n.pitches ])

    for c in flat.getElementsByClass(music21.harmony.ChordSymbol):
        chords.append(str(c.findFigure()))
        chord_kinds.append(c.chordKind or "unknown")

    tempo = flat.getElementsByClass(music21.tempo.MetronomeMark).first()
    if tempo:
        tempos.append(tempo.getQuarterBPM())

    key = flat.getElementsByClass(music21.key.KeySignature).first()
    if key:
        keys.append(key.sharps)

    time = flat.getElementsByClass(music21.meter.TimeSignature).first()
    if time:
        times.append(time.ratioString)

    lengths.append(len(flat.getElementsByClass(music21.stream.Measure)))
    voices.append(len(flat.getElementsByClass(music21.stream.Voice)))


# --- plotting ---
BAR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9,
})


def midi_name(m):
    return music21.pitch.Pitch(midi=int(m)).nameWithOctave


def key_name(sharps):
    return f"{music21.key.KeySignature(sharps).asKey('major').tonic.name} ({sharps:+d})"


def mark_extremes(ax, data, fmt=lambda v: f"{v:g}"):
    """Dashed lines + labels at the min and max of numeric data."""
    lo, hi = min(data), max(data)
    for v, label, ha in ((lo, "min", "left"), (hi, "max", "right")):
        ax.axvline(v, color=MUTED, linestyle="--", linewidth=1)
        ax.text(v, 0.97, f" {label} {fmt(v)} ", transform=ax.get_xaxis_transform(),
                ha=ha, va="top", color=INK, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID))
    ax.text(0.5, -0.22, f"n={len(data)}  mean={np.mean(data):.1f}  median={np.median(data):g}",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=8)


def mark_bar_extremes(ax, bars, counts):
    """Label the most and least common category in a bar chart."""
    counts = list(counts)
    hi, lo = int(np.argmax(counts)), int(np.argmin(counts))
    for i, label in ((hi, "max"), (lo, "min")):
        b = bars[i]
        ax.annotate(f"{label}: {counts[i]}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    color=INK, fontsize=8)
        if hi == lo:
            break


def bar_chart(ax, counter, title, xlabel, label_fn=str, sort_by_key=True):
    items = sorted(counter.items()) if sort_by_key else counter.most_common()
    labels = [label_fn(k) for k, _ in items]
    counts = [v for _, v in items]
    bars = ax.bar(range(len(counts)), counts, color=BAR, width=0.7)
    ax.set_xticks(range(len(counts)), labels, rotation=45 if len(labels) > 5 else 0, ha="right" if len(labels) > 5 else "center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_ylim(0, max(counts) * 1.18)
    mark_bar_extremes(ax, bars, counts)


fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle(f"Omnibook dataset — {len(lengths)} scores", fontsize=15, fontweight="bold")

# 1. pitch histogram (one bin per semitone)
ax = axes[0, 0]
ax.hist(pitches, bins=np.arange(min(pitches), max(pitches) + 2) - 0.5, color=BAR, edgecolor="white", linewidth=0.5)
ticks = [m for m in range(min(pitches), max(pitches) + 1) if m % 12 == 0] or [min(pitches)]
ax.set_xticks(ticks, [midi_name(m) for m in ticks])
ax.set_title("Pitch distribution")
ax.set_xlabel("pitch (MIDI)")
ax.set_ylabel("notes")
mark_extremes(ax, pitches, fmt=lambda v: f"{midi_name(v)} ({v})")

# 2. tempo histogram
ax = axes[0, 1]
ax.hist(tempos, bins=15, color=BAR, edgecolor="white", linewidth=0.5)
ax.set_title("Tempo")
ax.set_xlabel("quarter BPM")
ax.set_ylabel("scores")
mark_extremes(ax, tempos, fmt=lambda v: f"{v:g}")

# 3. song length histogram
ax = axes[0, 2]
ax.hist(lengths, bins=15, color=BAR, edgecolor="white", linewidth=0.5)
ax.set_title("Length")
ax.set_xlabel("measures")
ax.set_ylabel("scores")
mark_extremes(ax, lengths, fmt=lambda v: f"{v}")

# 4. key signatures
bar_chart(axes[1, 0], Counter(keys), "Key signature (major / sharps)", "key", label_fn=key_name)

# 5. time signatures
bar_chart(axes[1, 1], Counter(times), "Time signature", "meter")

# 6. pitches per note event (0 = rest)
bar_chart(axes[1, 2], Counter(pitch_chords), "Pitches per note event", "pitches (0 = rest)")
for b in axes[1, 2].patches:
    axes[1, 2].annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 14), textcoords="offset points", ha="center", color=MUTED, fontsize=8)

# 7. voices per score
bar_chart(axes[2, 0], Counter(voices), "Voices per score", "voice objects")

# 8. top 20 chord symbols
ax = axes[2, 1]
top = Counter(chords).most_common(20)[::-1]
bars = ax.barh([c for c, _ in top], [n for _, n in top], color=BAR, height=0.7)
ax.set_title(f"Top 20 chord symbols ({len(set(chords))} unique, {len(chords)} total)")
ax.set_xlabel("count")
ax.tick_params(axis="y", labelsize=8)
for b, (_, n), label in ((bars[-1], top[-1], "max"), (bars[0], top[0], "min of top 20")):
    ax.annotate(f" {label}: {n}", (b.get_width(), b.get_y() + b.get_height() / 2),
                va="center", color=INK, fontsize=8)
ax.set_xlim(0, top[-1][1] * 1.3)

# 9. chord quality
bar_chart(axes[2, 2], Counter(chord_kinds), "Chord quality", "kind", sort_by_key=False)

for ax in axes.flat:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
axes[2, 1].grid(axis="y", visible=False)
axes[2, 1].grid(axis="x", color=GRID, linewidth=0.6)

fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3)
# plt.show()
