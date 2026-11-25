import pathlib
import music21
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
from fractions import Fraction
import json

OMNIBOOK_DIR = pathlib.Path("./omnibook/")
OMNIBOOK_HEAD_LENGTH_DIR = pathlib.Path("./analysis/omnibook_head_lengths.json")

#--- tune class ---
class Tune:
    def __init__(self, name):
        self.title = name
        self.head_notes = []
        self.output_notes = []
    
    def load_from_file(self, path, head_length):
        score = music21.converter.parseFile(path)
        for i, n in enumerate(score.flatten().notesAndRests):
            note = NGramNote.from_music21(n)
            if note.dur <= 0: continue
            if n.measureNumber < head_length:
                self.head_notes.append(note)
            else:
                self.output_notes.append(note)

    @staticmethod
    def count_n_grams(notes, n):
        n_gram_dict = {}
        total_num = len(notes)
        for i in range(total_num - n):
            n_gram = notes[i:i+n]
            n_str = NGramNote.notes_to_gram_id(n_gram)
            if n_str == '': continue
            if n_str in n_gram_dict:
                n_gram_dict[n_str] += 1
            else: n_gram_dict[n_str] = 1
        return n_gram_dict
        
    @staticmethod
    def count_head_n_grams(tunes, n):
        n_gram_dict = {}
        for tune in tunes:
            tune_dict = Tune.count_n_grams(tune.head_notes, n)
            for gram_id, freq in tune_dict.items():
                if gram_id in n_gram_dict:
                    n_gram_dict[gram_id] += freq
                else:
                    n_gram_dict[gram_id] = freq
        return n_gram_dict
    @staticmethod
    def count_output_n_grams(tunes, n):
        n_gram_dict = {}
        for tune in tunes:
            tune_dict = Tune.count_n_grams(tune.output_notes, n)
            for gram_id, freq in tune_dict.items():
                if gram_id in n_gram_dict:
                    n_gram_dict[gram_id] += freq
                else:
                    n_gram_dict[gram_id] = freq
        return n_gram_dict

class NGramNote:
    """A note or rest: pitch is a MIDI number (-1 for a rest), dur is in quarter lengths."""
    REST = 100000

    def __init__(self, dur, pitch):
        self.dur = Fraction(dur).limit_denominator(48)
        self.pitch = pitch if isinstance(pitch, int) else int(pitch)

    @classmethod
    def from_music21(cls, note21: music21.note.GeneralNote):
        pitch = cls.REST if note21.isRest else note21.pitches[0].midi
        return cls(note21.duration.quarterLength, pitch)

    @property
    def is_rest(self):
        return self.pitch == self.REST

    def __str__(self):
        return f'{self.pitch},{self.dur}'

    def to_music21(self):
        ql = float(self.dur)
        if self.is_rest:
            return music21.note.Rest(quarterLength=ql)
        return music21.note.Note(midi=self.pitch + 72, quarterLength=ql)

    def normalized_pitch(self, b):
        return NGramNote(self.dur, self.pitch if self.is_rest else self.pitch - b)

    @staticmethod
    def music21_notes_to_gram_id(notes):
        return NGramNote.notes_to_gram_id([ NGramNote.from_music21(n) for n in notes ])

    @staticmethod
    def notes_to_gram_id(notes):
        orig_note = [ n for n in notes if not n.is_rest ]
        if len(orig_note) == 0: return ''
        orig_note = orig_note[0].pitch
        return '|'.join(str(n.normalized_pitch(orig_note)) for n in notes)

    @staticmethod
    def gram_id_to_notes(gram_id):
        notes = []
        for part in gram_id.split('|'):
            pitch, dur = part.split(',')
            notes.append(NGramNote(Fraction(dur), int(pitch)))
        return notes

    # --- legacy distance constants ---
    # Used only by the numba edit-distance kernel in connection_graph.py /
    # head_vs_output.py (an unbounded distance for MDS layout), not by calculate_diff.
    DIFF_INDEL = 0.7         # cost of inserting/deleting a note (scaled down for short notes)
    DIFF_MAX_SHIFT = 2       # also try transposing gram2 by up to ±this many semitones
    DIFF_OCTAVE_FACTOR = 0.3 # same pitch class in another octave costs this fraction

    # --- similarity score (0..1) ---
    # Relative weight of each component in calculate_diff (normalized to sum to 1).
    SIM_W_RHYTHM = 0.35
    SIM_W_INTERVAL = 0.35
    SIM_W_NOTE = 0.30

    @staticmethod
    def _align(a, b, sub, indel):
        """Normalized edit distance in [0, 1]. sub(x, y) and indel(x) must be in [0, 1],
        so the raw distance is at most max(len(a), len(b))."""
        m, k = len(a), len(b)
        if m == 0 and k == 0:
            return 0.0
        prev = [0.0] * (k + 1)
        for j in range(1, k + 1):
            prev[j] = prev[j-1] + indel(b[j-1])
        for i in range(1, m + 1):
            cur = [prev[0] + indel(a[i-1])] + [0.0] * k
            for j in range(1, k + 1):
                cur[j] = min(prev[j] + indel(a[i-1]),
                             cur[j-1] + indel(b[j-1]),
                             prev[j-1] + sub(a[i-1], b[j-1]))
            prev = cur
        return min(1.0, prev[k] / max(m, k))

    @staticmethod
    def _rhythm_similarity(gram1, gram2):
        """Durations and the note/rest pattern; pitch is ignored."""
        def sub(n1, n2):
            # one doubling/halving = 0.5, two or more = 1
            cost = min(abs(math.log2(float(n1.dur) / float(n2.dur))) / 2, 1.0)
            if n1.is_rest != n2.is_rest:
                cost += 0.5
            return min(cost, 1.0)
        # short notes (grace/passing tones) are cheaper to insert/delete
        indel = lambda n: min(1.0, 0.5 + float(n.dur))
        return 1.0 - NGramNote._align(gram1, gram2, sub, indel)

    @staticmethod
    def _intervals(gram):
        pitches = [n.pitch for n in gram if not n.is_rest]
        return [b - a for a, b in zip(pitches, pitches[1:])]

    @staticmethod
    def _interval_similarity(gram1, gram2):
        """Successive intervals between pitched notes (transposition-invariant)."""
        def sub(a, b):
            if a == b:
                return 0.0
            # any mismatch 0.3, plus size of the difference, plus 0.4 if the direction flips
            contour = 0.0 if (a > 0) - (a < 0) == (b > 0) - (b < 0) else 0.4
            return 0.3 + 0.3 * min(abs(a - b), 6) / 6 + contour
        return 1.0 - NGramNote._align(NGramNote._intervals(gram1),
                                      NGramNote._intervals(gram2), sub, lambda _: 1.0)

    @staticmethod
    def _note_similarity(gram1, gram2):
        """Note-by-note pitch match under the best transposition of gram2.

        Unlike intervals, a single wrong note costs once here rather than twice,
        and non-adjacent pitch relationships count.
        """
        p1 = [n.pitch for n in gram1 if not n.is_rest]
        p2 = [n.pitch for n in gram2 if not n.is_rest]
        if not p1 or not p2:
            return 1.0 if not p1 and not p2 else 0.0
        # the best transposition lines up at least one pair of notes
        shifts = {a - b for a in p1 for b in p2}
        def sub(a, b):
            if a == b:
                return 0.0
            if (a - b) % 12 == 0:  # same pitch class, other octave
                return 0.2
            return 0.5 + 0.5 * min(abs(a - b), 6) / 6
        return 1.0 - min(NGramNote._align(p1, [p + s for p in p2], sub, lambda _: 1.0)
                         for s in shifts)

    @staticmethod
    def calculate_diff(gram1, gram2):
        """Similarity between two grams: 1 = identical, 0 = nothing alike.

        Weighted mix of rhythmic, intervallic and note-wise similarity, each an
        alignment (edit distance) so an added/dropped note doesn't misalign
        everything after it. Transposition-invariant and symmetric.
        """
        w_r, w_i, w_n = NGramNote.SIM_W_RHYTHM, NGramNote.SIM_W_INTERVAL, NGramNote.SIM_W_NOTE
        score = (w_r * NGramNote._rhythm_similarity(gram1, gram2)
                 + w_i * NGramNote._interval_similarity(gram1, gram2)
                 + w_n * NGramNote._note_similarity(gram1, gram2)) / (w_r + w_i + w_n)
        return max(0.0, min(1.0, score))

# --- helper functions ---
def load_tunes():
    output = []
    head_lengths = load_head_lengths()
    for file in OMNIBOOK_DIR.iterdir():
        name = file.name.split('/')[-1].split('.')[0].replace('_', ' ')
        if head_lengths[name] == 0: 
            print(f'Skipped {name}')
            continue
        song = Tune(file.name)
        song.load_from_file(file, head_lengths[name])
        output.append(song)
    return output

def load_head_lengths():
    with open(OMNIBOOK_HEAD_LENGTH_DIR, 'r') as f:
        head_lengths = json.load(f)
    return head_lengths

# --- display functions ---
def _find_mscore():
    """Locate the MuseScore executable (env var MSCORE_PATH, PATH, or common install dirs)."""
    import shutil, glob
    candidates = [os.environ.get('MSCORE_PATH')]
    for name in ('mscore', 'MuseScore4', 'MuseScore4.exe', 'mscore4'):
        candidates.append(shutil.which(name))
    candidates += [
        '/Applications/MuseScore 4.app/Contents/MacOS/mscore',
        *glob.glob(r'C:\Program Files\MuseScore*\bin\MuseScore*.exe'),
        *glob.glob(r'C:\Program Files (x86)\MuseScore*\bin\MuseScore*.exe'),
        *glob.glob(os.path.expandvars(r'%LOCALAPPDATA%\Programs\MuseScore*\bin\MuseScore*.exe')),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        'MuseScore not found. Install MuseScore 4 or set the MSCORE_PATH environment '
        'variable to the full path of MuseScore4.exe.')

def _tuplet_group(dur):
    """Length of the tuplet group a duration belongs to (None for ordinary durations)."""
    if dur.denominator % 3 != 0:
        return None
    return Fraction(1, 2) if dur.denominator % 6 == 0 else Fraction(1)

def _render_all_ngrams(sorted_grams):
    """Render all n-grams in one Part as a 2x5 grid, return image."""
    env = music21.environment.Environment()
    mscore = _find_mscore()
    env['musescoreDirectPNGPath'] = mscore
    score = _build_ngram_score(sorted_grams)

    import subprocess, tempfile
    musicxml_fp = str(score.write('musicxml'))
    png_fp = musicxml_fp.replace('.musicxml', '.png').replace('.xml', '.png')
    subprocess.run([
        mscore,
        musicxml_fp, '-o', png_fp, '-r', '900'
    ], capture_output=True)
    # MuseScore may append a page number to the filename
    if not os.path.exists(png_fp):
        base, ext = os.path.splitext(png_fp)
        png_fp = f'{base}-1{ext}'
    return mpimg.imread(png_fp)


def _build_ngram_score(sorted_grams):
    """Build a music21 Score laying out the given (gram_id, freq) pairs, 4 measures per system."""
    # Every gram occupies the same number of 4/4 measures (K), padded with rests,
    # so grams of any length keep a uniform width.
    BAR_QL = 4
    totals = [sum(n.dur for n in NGramNote.gram_id_to_notes(g)) for g, _ in sorted_grams]
    K = max(1, math.ceil((float(max(totals)) + 0.5) / BAR_QL))  # +0.5: room for the end gap
    per_row = max(1, 4 // K)  # grams per system, keeps rows from wrapping mid-gram
    part = music21.stream.Part()
    measure_no = 1

    for i, (gram_id, freq) in enumerate(sorted_grams):
        notes = NGramNote.gram_id_to_notes(gram_id)

        cell = music21.stream.Stream()
        t = Fraction(0)       # running offset in quarter lengths
        open_group = None     # tuplet group length of the tuplet run in progress

        def close_tuplet():
            # A gram can cut a tuplet short. Fill the gap with an invisible rest so
            # the notes after it stay on the grid (otherwise MuseScore fails to load it).
            nonlocal t, open_group
            if open_group and t % open_group:
                gap = open_group - t % open_group
                filler = music21.note.Rest(quarterLength=float(gap))
                filler.style.hideObjectOnPrint = True
                cell.append(filler)
                t += gap
            open_group = None

        for ngram_note in notes:
            if ngram_note.dur <= 0:
                continue
            group = _tuplet_group(ngram_note.dur)
            if group != open_group:
                close_tuplet()
                open_group = group
            cell.append(ngram_note.to_music21())
            t += ngram_note.dur
        close_tuplet()
        # pad with rests up to exactly K measures
        pad = K * BAR_QL - float(cell.duration.quarterLength)
        if pad > 0:
            cell.append(music21.note.Rest(quarterLength=pad))

        cell.insert(0, music21.meter.TimeSignature(f'{BAR_QL}/4'))
        measures = list(cell.makeMeasures().makeTies(inPlace=False).getElementsByClass('Measure'))

        for j, measure in enumerate(measures):
            measure.number = measure_no
            measure_no += 1
            # time signature only on the very first measure, hidden
            for ts in list(measure.getElementsByClass('TimeSignature')):
                if i == 0 and j == 0:
                    ts.style.hideObjectOnPrint = True
                else:
                    measure.remove(ts)
            if j == 0:
                if i % per_row == 0:
                    measure.insert(0, music21.layout.SystemLayout(isNew=True))
                # Frequency label above the first measure of the gram
                te = music21.expressions.TextExpression(f'×{freq}')
                te.placement = 'above'
                measure.insert(0, te)
            if j == len(measures) - 1:
                measure.rightBarline = music21.bar.Barline('double')
            part.append(measure)

    score = music21.stream.Score()
    # Increase notation size (default ~7mm per 40 tenths)
    sl = music21.layout.ScoreLayout()
    sl.scalingMillimeters = 12 if K == 1 else 7  # shrink so multi-measure grams still fit on one page
    sl.scalingTenths = 40
    score.insert(0, sl)
    score.append(part)
    return score


def _gram_label(gram_id):
    names = []
    for n in NGramNote.gram_id_to_notes(gram_id):
        names.append('R' if n.is_rest else music21.pitch.Pitch(midi=n.pitch).nameWithOctave)
    return ' '.join(names)

def display_n_gram_dict(n_gram_dict):
    sorted_grams = sorted(n_gram_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    n = len(sorted_grams)
    if n == 0:
        return

    labels = [_gram_label(g[0]) for g in sorted_grams]
    freqs = [g[1] for g in sorted_grams]

    # Render all notation in a single MuseScore invocation
    notation_img = _render_all_ngrams(sorted_grams)

    fig = plt.figure(figsize=(8, 5))
    left, right = fig.subfigures(1, 2, width_ratios=[1, 1.5])

    # Left: frequency bar chart
    ax_bar = left.subplots()
    bars = ax_bar.barh(range(n), freqs, color='#2a78d6', edgecolor='white', height=0.6)
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(labels, fontfamily='monospace', fontsize=8)
    ax_bar.set_xlabel('Frequency')
    ax_bar.set_title('Top 10 N-Gram Frequencies')
    ax_bar.invert_yaxis()
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    for bar, f in zip(bars, freqs):
        ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(f), va='center', fontsize=8)

    # Right: single notation image from music21
    ax_img = right.subplots()
    ax_img.imshow(notation_img)
    ax_img.axis('off')

    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tunes = load_tunes()
    n_gram_dict = Tune.count_head_n_grams(tunes, 8)
    display_n_gram_dict(n_gram_dict)
