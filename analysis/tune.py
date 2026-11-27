import pathlib
import music21
from fractions import Fraction
import json
import numpy as np
import kernels

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
        self.load_from_score(score, head_length)

    def load_from_score(self, score, head_length):
        for i, n in enumerate(score.flatten().notesAndRests):
            note = Note.from_music21(n)
            if note.dur <= 0: continue
            if n.measureNumber < head_length:
                self.head_notes.append(note)
            else:
                self.output_notes.append(note)

    @staticmethod
    def count_n_grams(notes, n):
        n_gram_dict = {}
        total_num = len(notes)
        for i in range(total_num - n + 1):
            n_gram = notes[i:i+n]
            n_str = Note.notes_to_gram_id(n_gram)
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

class Note:
    """A note or rest: pitch is a MIDI number (Note.REST for a rest), dur is in quarter lengths."""
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
        return Note(self.dur, self.pitch if self.is_rest else self.pitch - b)

    @staticmethod
    def music21_notes_to_gram_id(notes):
        return Note.notes_to_gram_id([ Note.from_music21(n) for n in notes ])

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
            notes.append(Note(Fraction(dur), int(pitch)))
        return notes

    @staticmethod
    def contour_elements(notes):
        """Intervals between consecutive pitched notes as kernel elements (interval, direction, indel cost)."""
        pitches = [ n.pitch for n in notes if not n.is_rest ]
        return np.array([ (b - a, np.sign(b - a), 1) for a, b in zip(pitches, pitches[1:]) ], np.float32).reshape(-1, 3)

    @staticmethod
    def rhythm_elements(notes):
        """Durations as kernel elements (log2 duration, is_rest, indel cost); short notes are cheaper to insert/delete."""
        return np.array([ (np.log2(float(n.dur)), n.is_rest, min(0.5 + float(n.dur) / 2, 1)) for n in notes ], np.float32).reshape(-1, 3)

    @staticmethod
    def encode_grams(gram_ids, elements):
        """Padded (V, L, 3) element array and (V,) lengths for the kernels."""
        rows = [ elements(Note.gram_id_to_notes(g)) for g in gram_ids ]
        lengths = np.array([ len(r) for r in rows ], np.int64)
        E = np.zeros((len(rows), max(lengths.max(), 1), 3), np.float32)
        for i, r in enumerate(rows): E[i, :len(r)] = r
        return E, lengths

    @staticmethod
    def calculate_intervallic_difference(gram1, gram2):
        """Returns value from 0 to 1 with 0 being completely distinct and 1 being identical. Measures the shape of musical contour (transposition invariant, ignores rhythm). Grams are gram id strings."""
        e1, e2 = (Note.contour_elements(Note.gram_id_to_notes(g)) for g in (gram1, gram2))
        return kernels.pair_sim(e1, e2, kernels.CONTOUR_W)

    @staticmethod
    def calculate_rhythmic_difference(gram1, gram2):
        """Returns value from 0 to 1 with 0 being completely distinct and 1 being identical. Measures rhythm (durations and rests, ignores pitch). Grams are gram id strings."""
        e1, e2 = (Note.rhythm_elements(Note.gram_id_to_notes(g)) for g in (gram1, gram2))
        return kernels.pair_sim(e1, e2, kernels.RHYTHM_W)

    @staticmethod
    def calculate_difference(gram1, gram2):
        """Returns value from 0 to 1 with 0 being completely distinct and 1 being identical"""
        return Note.calculate_intervallic_difference(gram1, gram2) * Note.calculate_rhythmic_difference(gram1, gram2)

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
