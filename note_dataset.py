import torch
import numpy as np
from collections import OrderedDict

def onehot(idx,n_values):
    onehot = np.zeros(n_values)
    onehot[idx]=1
    return onehot

def model_format_to_noteseq(x):
    """
    Converts a model input format to a note sequence.
    """
    note_sequence = []
    n_timesteps, n_pitches = next(iter(x.values())).shape
    for timestep_index in range(n_timesteps):
        if np.argmax(x["type"][timestep_index])==0:
            note_sequence.append({"pitch":np.argmax(x["pitch"][timestep_index]),
                                "start":np.argmax(x["onset"][timestep_index]),
                                "end":np.argmax(x["onset"][timestep_index])+np.argmax(x["duration"][timestep_index])})
    return note_sequence
                
def noteseq_to_model_format(note_sequence, n_pitches,n_timesteps, sequence_length):
    """
    Converts a note sequence to a model input format.
    ATTENTION: Only takes every other note!!
    """
    note_type=[]
    note_pitch=[]
    note_onset=[]
    note_duration=[]
    for step_index in range(sequence_length):
        if step_index < len(note_sequence) and note_sequence[step_index]["start"]%2==0:
            # type section
            note_type.append(onehot(0,2))
            # pitch section
            note_pitch.append(onehot(note_sequence[step_index]["pitch"],n_pitches))
            # onset section
            note_onset.append(onehot(note_sequence[step_index]["start"],n_timesteps))
            # duration section
            duration = note_sequence[step_index]["end"]-note_sequence[step_index]["start"]
            note_duration.append(onehot(duration, n_timesteps))
        else:
            note_type.append(onehot(1,2))
            # uniform distribution
            note_pitch.append(np.ones(n_pitches)/n_pitches)
            note_onset.append(np.ones(n_timesteps)/n_timesteps)
            note_duration.append(np.ones(n_timesteps)/n_timesteps)
    seq = OrderedDict()
    seq["type"] = np.stack(note_type)
    seq["pitch"] = np.stack(note_pitch)
    seq["onset"] = np.stack(note_onset)
    seq["duration"] = np.stack(note_duration)
    return seq

class NoteDataset(torch.utils.data.Dataset):
    def __init__(self, prepared_data_path, crop_size=None):
        self.crop_size = crop_size
        self.load_data(prepared_data_path)
        # filter away noteseq of length 0
        self.data = [example for example in self.data if len(example["note_seq"]) > 0]

    def load_data(self, path):
        self.data = torch.load(path)

    def save_data(self, path):
        torch.save(self.data,path)

    def get_token_sections(self):
        example_seq = self[0]["seq"]
        token_sections = []
        for key in example_seq.keys():
            token_sections.append({"label":key,"n_channels":example_seq[key].shape[-1]})
        assert token_sections[0]["label"] == "type"
        return token_sections

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

        seq = noteseq_to_model_format(note_seq,36,64,128)
        
        return {"seq":seq}
        

    def __len__(self):
        return len(self.data)