import pretty_midi
import numpy as np
#  This script is to convert piano roll to pretty_midi object, with extra input on note onsets.
#  Consecutive pitches that are the same are no longer "assumed" to be held as a long note.
#
#  Original function comes from pypianoroll: 
#  https://github.com/salu133445/pypianoroll/blob/4d87eba9fc3dca0b353c85e303caf0d25ccd2e29/pypianoroll/multitrack.py#L904


def to_pretty_midi(pr, ctrl=None, constant_tempo=None, constant_velocity=100):
    '''
    Parameters
    ----------
    pr    : NumPy array of size (t, 128)
    ctrl  : list of length t with - 
              - binary values 0 and 1, where 1 denotes a note onset (for monophonic)
              - 0 and pitch values, where pitch values denote a note onset (for polyphonic)
        
    Returns
    -------
    pm    : `pretty_midi.PrettyMIDI` object
        The converted :class:`pretty_midi.PrettyMIDI` instance.
    '''
    beat_resolution = 4
    pm = pretty_midi.PrettyMIDI()

    if constant_tempo is None:
        constant_tempo = 128
    time_step_size = 60. / constant_tempo / beat_resolution

    instrument = pretty_midi.Instrument(program=0, is_drum=False, name="test")
    clipped = pr.astype(np.uint8)
    binarized = (clipped > 0)
    padded = np.pad(binarized, ((1, 1), (0, 0)), 'constant')
    diff = np.diff(padded.astype(np.int8), axis=0)
    
    positives = np.nonzero((diff > 0).T)
    pitches = positives[0]
    note_ons = positives[1]
    note_on_times = time_step_size * note_ons
    note_offs = np.nonzero((diff < 0).T)[1]
    note_off_times = time_step_size * note_offs
    
    if ctrl is None:
        for idx, pitch in enumerate(pitches):
            velocity = np.mean(clipped[note_ons[idx]:note_offs[idx], pitch])
            note = pretty_midi.Note(
                velocity=int(velocity), pitch=pitch,
                start=note_on_times[idx], end=note_off_times[idx])
            instrument.notes.append(note)
    
    else:
        pairs = []
        for idx, pitch in enumerate(pitches):
            note_on, note_off = note_ons[idx], note_offs[idx]
            true_ons = ctrl[note_ons[idx]:note_offs[idx]]
            on_idx = [i for i in range(len(true_ons)) if true_ons[i] == 1]  # if polyphonic, change 1 to pitch value
            on_idx.pop(0)  # remove 1st onset token

            cur_note_on = note_on
            while on_idx:
                cur_note_off = note_on + on_idx[0]
                pairs.append((pitch, cur_note_on, cur_note_off))
                cur_note_on = cur_note_off
                on_idx.pop(0)
            pairs.append((pitch, cur_note_on, note_off))   
        
        for idx, p in enumerate(pairs):
            pitch, start, end = p
            velocity = np.mean(clipped[start:end, pitch])
            note = pretty_midi.Note(
                velocity=int(velocity), pitch=pitch,
                start=start*time_step_size, end=end*time_step_size)
            instrument.notes.append(note)

    instrument.notes.sort(key=lambda x: x.start)
    pm.instruments.append(instrument)
    
    return pm

# #  Usage
# #  Assume t to be a list of pitch values, with token 128 denote holding the pitch
# ctrl = []
# for k in t.cpu().numpy().squeeze():
#     if k == 128: ctrl.append(0)
#     else: ctrl.append(1)

# pm = to_pretty_midi(pr, ctrl)
# audio = pm.fluidsynth()
# Audio(audio, rate=44100)