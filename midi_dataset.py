#%%
import glob
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import soundfile as sf
import torch
from IPython.display import Audio, display
from PIL import Image
from tqdm import tqdm

from chiptune import chiptunes_synthesize
from piano_roll_to_pretty_midi import to_pretty_midi
from util import (crop_augment_piano_roll, crop_roll_128_to_88, get_onsets,
                  pad_roll_88_to_128)


def transpose(midi_data, offset):
    for instrument in midi_data.instruments:
        # Don't want to shift drum notes
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += offset
    return midi_data

def change_tempo(midi_data, factor):
    # Get the length of the MIDI file
    length = midi_data.get_end_time()
    midi_data.adjust_times([0, length], [0, length*factor])
    return midi_data

def has_simulateneous_neighbouring_notes(roll):
    roll = (roll>0).astype(np.int32)
    padded_roll = np.pad(roll,((1,1),(0,0)),"constant",constant_values=0)
    two_note_sum = padded_roll + np.roll(padded_roll,1,0)
    return (np.max(two_note_sum)>1)

def piano_roll_to_note_seq(piano_roll):
    note_seq=[]
    for pitch in range(piano_roll.shape[0]):
        current_note = None
        for time in range(piano_roll.shape[1]):
            if piano_roll[pitch,time] > 0:
                if current_note is None:
                    current_note = {"pitch":pitch,"start":time,"end":time+1,"velocity":piano_roll[pitch,time]}
                else:
                    current_note["end"] += 1
            else:
                if current_note is not None:
                    note_seq.append(current_note)
                    current_note = None
    note_seq = sorted(note_seq,key=lambda x: x["start"])
    return note_seq
    
class MidiDataset(torch.utils.data.Dataset):
    def __init__(self,midi_filepaths=None, prepared_data_path = None, crop_size=None,downsample_factor=1, only_88_keys=True, mode="piano_roll"):
        self.mode = mode
        self.crop_size=crop_size
        self.downsample_factor = downsample_factor
        self.N_STEPS=64

        if prepared_data_path is not None:
            self.data = torch.load(prepared_data_path)
        else:
            self.fps = glob.glob(midi_filepaths,recursive=True)
            self.data = []
            for fp in tqdm(self.fps):
                try:
                    piano_roll = self.load_piano_roll(fp)
                except:
                    continue
                if only_88_keys:
                    piano_roll = crop_roll_128_to_88(piano_roll)
                piano_rolls = self.chunk_piano_roll(piano_roll)

                if len(piano_rolls) > 0:
                    for piano_roll in piano_rolls:
                        self.data.append({"piano_roll":piano_roll,"caption":self.fp_to_caption(fp), "filepath":fp})

        print("Loaded {} examples".format(len(self.data)))
        
        # make sure all piano rolls are of size N_STEPS
        self.data = [example for example in self.data if example["piano_roll"].shape[1] == self.N_STEPS]

        print("Filtered to {} examples because of size".format(len(self.data)))

        if self.downsample_factor != 1:
            # downsample all piano rolls
            for i in range(len(self.data)):
                self.data[i]["piano_roll"] = self.data[i]["piano_roll"][:,::self.downsample_factor]

        # remove all empty piano rolls
        self.data = [ example for example in self.data if example["piano_roll"].sum() > 0]

        print("Filtered to {} examples because of empty".format(len(self.data)))

        # filter away rolls with less than 8 onsets
        self.data = [example for example in self.data if np.sum(get_onsets(example["piano_roll"][None,...]))>7]

        print("Filtered to {} examples because of too few onsets".format(len(self.data)))
        
        # filter away rolls with neighbouring notes being played simultaneously
        self.data = [example for example in self.data if not has_simulateneous_neighbouring_notes(example["piano_roll"])]
       
        print("Filtered to {} examples because of neighbouring notes".format(len(self.data)))
        
        new_data = []
        seen = set()
        # filter away non unique piano rolls
        for i in range(len(self.data)):
            if self.data[i]["piano_roll"].tostring() in seen:
                continue
            else:
                new_data.append(self.data[i])
                seen.add(self.data[i]["piano_roll"].tostring())
        self.data = new_data

        print("Filtered to {} after removing duplicates".format(len(self.data)))

        if self.mode == "note_seq":
            for i in tqdm(range(len(self.data))):
                self.data[i]["dim_source_roll"] = self.data[i]["piano_roll"].shape
                self.data[i]["note_seq"] = piano_roll_to_note_seq(self.data[i]["piano_roll"])
                self.data[i]["piano_roll"] = None
 
       
    def __getitem__(self, idx):
        example = self.data[idx]

        roll = example["piano_roll"].copy()

        # random crop
        if self.crop_size is not None:
            roll = crop_augment_piano_roll(roll,self.crop_size)
        
        return {"piano_roll":roll,"caption":example["caption"]}

    def save_data(self, path):
        torch.save(self.data,path)

    def chunk_piano_roll(self,full_piano_roll):

        if full_piano_roll.shape[1] < self.N_STEPS:
            return []

        # trim end to multiple of N_STEPS
        full_piano_roll = full_piano_roll[:,:full_piano_roll.shape[1]//self.N_STEPS*self.N_STEPS]

        # split into 64 step chunks
        piano_rolls = np.split(full_piano_roll,full_piano_roll.shape[1]//self.N_STEPS,axis=1)
        
        # filter away empty chunks
        piano_rolls = [piano_roll for piano_roll in piano_rolls if np.sum(piano_roll) > 0]

        return piano_rolls


    def __len__(self):
        return len(self.data)

    def fp_to_caption(self,fp):
        split_fp = fp.split("/")

        file_name = split_fp[-1]

        game_name = split_fp[-2]

        title = file_name.split("-")[-1]

        title = title.split(".")[0]
        # remove numbers
        title = "".join([i for i in title if not i.isdigit()])

        caption = f"{game_name} {title}"

        return caption


    def load_piano_roll(self,fp):
        midi = pretty_midi.PrettyMIDI(fp)

        #check 4 time
        assert len(midi.time_signature_changes) < 2
        if len(midi.time_signature_changes) == 1:
            assert midi.time_signature_changes[0].numerator == 4

        beat_times=midi.get_beats()

        beat_ticks = [midi.time_to_tick(time) for time in beat_times]

        # get beat length
        quarter_length = beat_ticks[1]-beat_ticks[0]

        # check that beats are all the same length
        for i in range(len(beat_ticks)-1):
            assert beat_ticks[i+1]-beat_ticks[i] == quarter_length

        # check that beats is a multiple of 4
        assert quarter_length%4 == 0

        steps_per_beat = 4

        # get 16th note length
        step_length = quarter_length//steps_per_beat
        
        last_end = 0
        # quantize midi to nearest 16th note
        for instrument in midi.instruments:
            for note in instrument.notes:
                note.start = midi.tick_to_time(step_length*(midi.time_to_tick(note.start)//step_length))
                note.end = midi.tick_to_time(step_length*(midi.time_to_tick(note.end)//step_length))
                if note.end > last_end:
                    last_end = note.end

        for instrument in midi.instruments:
            if instrument.is_drum:
                instrument.notes = []

        # get time of first event

        # convert to ticks
        first_onset_ticks = midi.time_to_tick(midi.get_onsets()[0])

        # get beat of first event
        first_beat = first_onset_ticks//quarter_length

        # get time of first beat
        first_beat_time = midi.tick_to_time(first_beat*quarter_length)

        n_beats=len(beat_times)
        n_steps = n_beats*steps_per_beat


        sampling_steps_per_beat = steps_per_beat

        # get 16th note in seconds
        sampling_step_time = midi.tick_to_time(quarter_length/sampling_steps_per_beat)

        # get piano roll
        piano_roll = midi.get_piano_roll(fs=1/sampling_step_time,times=np.arange(first_beat_time,last_end+midi.tick_to_time(quarter_length),sampling_step_time))
        
        return piano_roll


class NoteSeqDataset(torch.utils.data.Dataset):
    def __init__(self, prepared_data_path, crop_size=None):
        self.crop_size = crop_size
        self.load_data(prepared_data_path)
        # filter away noteseq of length 0
        self.data = [example for example in self.data if len(example["note_seq"]) > 0]
        

    def load_data(self, path):
        self.data = torch.load(path)

    def __getitem__(self, idx):
        example = self.data[idx]
        note_seq = example["note_seq"].copy()

        # get min and max pitch
        min_pitch = np.min([note["pitch"] for note in note_seq])
        max_pitch = np.max([note["pitch"] for note in note_seq])

        low = min(max_pitch-self.crop_size,min_pitch)
        high = max(max_pitch-self.crop_size,min_pitch)

        if low==high:
            start_pitch = low
        else:
            start_pitch = np.random.randint(low=low,high=high)
     
        end_pitch = start_pitch+self.crop_size

        # remove all notes outside of range and shift pitch to start at 0
        note_seq = [note for note in note_seq if note["pitch"] >= start_pitch and note["pitch"] < end_pitch]
        for note in note_seq:
            note["pitch"] -= start_pitch
        
        return {**example,"note_seq":note_seq}
        

    def __len__(self):
        return len(self.data)
# %%
