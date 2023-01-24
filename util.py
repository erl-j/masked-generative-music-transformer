import torch
import numpy as np
def crop_roll_128_to_88(full_piano_roll):
    assert full_piano_roll.shape[0] == 128
    # piano range
    full_piano_roll = full_piano_roll[21:109,:]
    return full_piano_roll

def pad_roll_88_to_128(piano_roll):
    assert piano_roll.shape[0] == 88
    # pad to piano range
    piano_roll = torch.nn.functional.pad(piano_roll,(0,0,21,21))
    return piano_roll


def crop_augment_piano_roll(piano_roll,crop_size):

    n_pitches , n_timesteps = piano_roll.shape

    pad_size = crop_size

    piano_roll = np.pad(piano_roll,((pad_size,pad_size),(0,0)),'constant',constant_values=0)

    time_sum=np.sum(piano_roll>0,axis=-1)

    min_pitch = np.min(np.where(time_sum>0)[0])
    max_pitch = np.max(np.where(time_sum>0)[0])

    low = min(max_pitch-crop_size,min_pitch)
    high = max(max_pitch-crop_size,min_pitch)

    if low==high:
        start_pitch = low
    else:
        start_pitch = np.random.randint(low=low,high=high)
        
    end_pitch = start_pitch+crop_size

    piano_roll  = piano_roll[start_pitch:end_pitch,:]

    return piano_roll


def get_onsets(rolls):
    # batch, pitch, time
    return (np.diff(rolls,prepend=0,axis=-1)>0).astype(np.int32)


def noteseq_to_roll(note_seq,n_pitches,n_timesteps):
    roll = np.zeros((n_pitches,n_timesteps))
    for note in note_seq:
        roll[note["pitch"],note["start"]:note["end"]] = note["velocity"]
    return roll

