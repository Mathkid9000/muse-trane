import pathlib
import music21
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
from fractions import Fraction

OMNIBOOK_DIR = pathlib.Path("./omnibook/")

#--- tune class ---
class Tune:
    def __init__(self, name):
        self.title = name
        self.head_notes = []
        self.output_notes = []
    
    def load_from_file(self, path):
        score = music21.converter.parseFile(path)
        for i, n in enumerate(score.flatten().notesAndRests):
            note = NGramNote.from_music21(n)
            if note.dur <= 0: continue
            if n.measureNumber < 16:
                self.head_notes.append(note)
            else:
                self.output_notes.append(note)
    
    def _count_head_n_grams(self, n, n_gram_dict):
        total_num = len(self.head_notes)
        for i in range(total_num - n):
            n_gram = self.head_notes[i:i+n]
            # if len([ n for n in n_gram if n.dur.denominator % 3 == 0 ]) % 3 != 0:
            #     continue
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
            tune_dict = tune._count_head_n_grams(n, {})
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

    # --- similarity (weighted edit distance) ---
    # Tuning constants. connection_graph.py has a numba copy of this algorithm that
    # reads these same values, so changing them here changes the graph too.
    DIFF_INDEL = 0.7         # cost of inserting/deleting a note (scaled down for short notes)
    DIFF_MAX_SHIFT = 2       # also try transposing gram2 by up to ±this many semitones
    DIFF_OCTAVE_FACTOR = 0.3 # same pitch class in another octave costs this fraction

    @staticmethod
    def _pitch_cost(a, b):
        """a, b: pitches relative to the gram's first note."""
        if a == b:
            return 0.0
        cost = min(abs(a - b), 12) / 12
        if (a - b) % 12 == 0:  # octave displacement: same pitch class
            cost *= NGramNote.DIFF_OCTAVE_FACTOR
        return cost

    @staticmethod
    def _event_cost(n1, n2, shift=0):
        """Cost of substituting note n1 for n2 (n2's pitch transposed by `shift`)."""
        # ratio-based: one doubling/halving = 0.25, capped at 0.5
        rhythm = min(abs(math.log2(float(n1.dur) / float(n2.dur))) / 4, 0.5)
        if n1.is_rest != n2.is_rest:
            return 1.0 + rhythm
        if n1.is_rest:
            return rhythm
        return NGramNote._pitch_cost(n1.pitch, n2.pitch + shift) + rhythm

    @staticmethod
    def _indel_cost(n):
        """Inserting/deleting a note; short notes (grace/passing tones) are cheaper."""
        return NGramNote.DIFF_INDEL * min(1.0, 0.5 + float(n.dur))

    @staticmethod
    def _edit_distance(gram1, gram2, shift=0):
        m, k = len(gram1), len(gram2)
        D = [[0.0] * (k + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            D[i][0] = D[i-1][0] + NGramNote._indel_cost(gram1[i-1])
        for j in range(1, k + 1):
            D[0][j] = D[0][j-1] + NGramNote._indel_cost(gram2[j-1])
        for i in range(1, m + 1):
            for j in range(1, k + 1):
                D[i][j] = min(D[i-1][j] + NGramNote._indel_cost(gram1[i-1]),
                              D[i][j-1] + NGramNote._indel_cost(gram2[j-1]),
                              D[i-1][j-1] + NGramNote._event_cost(gram1[i-1], gram2[j-1], shift))
        return D[m][k]

    @staticmethod
    def calculate_diff(gram1, gram2):
        """Musical distance between two grams (0 = identical, larger = more different).

        Weighted edit distance: notes may be substituted (cost from pitch + rhythm),
        or inserted/deleted (so an added passing tone or a split note is cheap rather
        than misaligning everything after it). Pitches are relative to each gram's
        first note; trying small transpositions of gram2 means a single early interval
        change isn't charged again at every later note. Symmetric.
        """
        return min(NGramNote._edit_distance(gram1, gram2, s)
                   for s in range(-NGramNote.DIFF_MAX_SHIFT, NGramNote.DIFF_MAX_SHIFT + 1))

# --- helper functions ---
def load_tunes():
    output = []
    for file in OMNIBOOK_DIR.iterdir():
        song = Tune(file.name)
        song.load_from_file(file)
        output.append(song)
    return output
    
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
