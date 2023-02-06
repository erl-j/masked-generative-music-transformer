import torch
import numpy as np
from collections import OrderedDict

def onehot(idx,n_values):
    onehot = np.zeros(n_values)
    onehot[idx]=1
    return onehot

def onehot_to_index(onehot):
    return np.argmax(onehot)



def model_format_to_noteseq(x):
    """
    Converts a model input format to a note sequence.
    """
    note_sequence = []
    n_timesteps, _ = next(iter(x.values())).shape
    for t in range(n_timesteps):
        note_type = onehot_to_index(x["type"][t])
        if note_type == 0:
            note_octave = onehot_to_index(x["octave"][t])
            note_chroma = onehot_to_index(x["chroma"][t])
            note_pitch = note_octave*12+note_chroma

            note_onset_32 = onehot_to_index(x["onset_32"][t])
            note_onset_16 = onehot_to_index(x["onset_16"][t])
            note_onset_8 = onehot_to_index(x["onset_8"][t])
            note_onset_4 = onehot_to_index(x["onset_4"][t])
            note_onset_2 = onehot_to_index(x["onset_2"][t])
            note_onset_1 = onehot_to_index(x["onset_1"][t])

            note_duration_32 = onehot_to_index(x["duration_32"][t])
            note_duration_16 = onehot_to_index(x["duration_16"][t])
            note_duration_8 = onehot_to_index(x["duration_8"][t])
            note_duration_4 = onehot_to_index(x["duration_4"][t])
            note_duration_2 = onehot_to_index(x["duration_2"][t])
            note_duration_1 = onehot_to_index(x["duration_1"][t])

            note_onset = note_onset_32*32+note_onset_16*16+note_onset_8*8+note_onset_4*4+note_onset_2*2+note_onset_1*1
            note_duration = note_duration_32*32+note_duration_16*16+note_duration_8*8+note_duration_4*4+note_duration_2*2+note_duration_1*1

            note_sequence.append({"pitch":note_pitch,"start":note_onset,"end":note_onset+note_duration})
        
    return note_sequence
        
def noteseq_to_model_format(note_sequence, n_pitches, n_timesteps, sequence_length):
    """
    Converts a note sequence to a model input format.
    ATTENTION: Only takes every other note!!
    """
    note_type=[]
    note_chroma=[]
    note_octave=[]

    note_onset_32=[]
    note_onset_16=[]
    note_onset_8=[]
    note_onset_4=[]
    note_onset_2=[]
    note_onset_1=[]

    note_duration_32=[]
    note_duration_16=[]
    note_duration_8=[]
    note_duration_4=[]
    note_duration_2=[]
    note_duration_1=[]

    n_octaves = int(np.ceil(n_pitches/12))

    for step_index in range(sequence_length):
        if step_index < len(note_sequence) and note_sequence[step_index]["start"]%2==0:
            # type section
            note_type.append(onehot(0,2))
            # pitch section
            octave = int(note_sequence[step_index]["pitch"]//12)
            chroma = int(note_sequence[step_index]["pitch"]%12)

            note_octave.append(onehot(octave,n_octaves))
            note_chroma.append(onehot(chroma,12))

            onset = note_sequence[step_index]["start"]

            onset_binary = np.binary_repr(onset, width=6)

            note_onset_32.append(onehot(int(onset_binary[0]), 2))
            note_onset_16.append(onehot(int(onset_binary[1]), 2))
            note_onset_8.append(onehot(int(onset_binary[2]), 2))
            note_onset_4.append(onehot(int(onset_binary[3]), 2))
            note_onset_2.append(onehot(int(onset_binary[4]), 2))
            note_onset_1.append(onehot(int(onset_binary[5]), 2))


            duration = note_sequence[step_index]["end"]-note_sequence[step_index]["start"]

            duration_binary = np.binary_repr(duration, width=6)

            note_duration_32.append(onehot(int(duration_binary[0]), 2))
            note_duration_16.append(onehot(int(duration_binary[1]), 2))
            note_duration_8.append(onehot(int(duration_binary[2]), 2))
            note_duration_4.append(onehot(int(duration_binary[3]), 2))
            note_duration_2.append(onehot(int(duration_binary[4]), 2))
            note_duration_1.append(onehot(int(duration_binary[5]), 2))

        else:

            note_type.append(onehot(1,2))
            note_octave.append(onehot(0,n_octaves)*0+1/n_octaves)
            note_chroma.append(onehot(0,12)*0+1/12)

            note_onset_32.append(onehot(0,2)*0+1/2)
            note_onset_16.append(onehot(0,2)*0+1/2)
            note_onset_8.append(onehot(0,2)*0+1/2)
            note_onset_4.append(onehot(0,2)*0+1/2)
            note_onset_2.append(onehot(0,2)*0+1/2)
            note_onset_1.append(onehot(0,2)*0+1/2)

            note_duration_32.append(onehot(0,2)*0+1/2)
            note_duration_16.append(onehot(0,2)*0+1/2)
            note_duration_8.append(onehot(0,2)*0+1/2)
            note_duration_4.append(onehot(0,2)*0+1/2)
            note_duration_2.append(onehot(0,2)*0+1/2)
            note_duration_1.append(onehot(0,2)*0+1/2)
            
    seq = OrderedDict()
    seq["type"] = np.stack(note_type)
    seq["chroma"] = np.stack(note_chroma)
    seq["octave"] = np.stack(note_octave)
    seq["onset_32"] = np.stack(note_onset_32)
    seq["onset_16"] = np.stack(note_onset_16)
    seq["onset_8"] = np.stack(note_onset_8)
    seq["onset_4"] = np.stack(note_onset_4)
    seq["onset_2"] = np.stack(note_onset_2)
    seq["onset_1"] = np.stack(note_onset_1)
    seq["duration_32"] = np.stack(note_duration_32)
    seq["duration_16"] = np.stack(note_duration_16)
    seq["duration_8"] = np.stack(note_duration_8)
    seq["duration_4"] = np.stack(note_duration_4)
    seq["duration_2"] = np.stack(note_duration_2)
    seq["duration_1"] = np.stack(note_duration_1)

    seq["type"]=np.concatenate((seq["type"],onehot(1,2)*np.ones((sequence_length-seq["type"].shape[0],2))),axis=0)
    
    return seq


# def model_format_to_noteseq(x):
#     """
#     Converts a model input format to a note sequence.
#     """
#     note_sequence = []
#     n_timesteps, n_pitches = next(iter(x.values())).shape
#     for timestep_index in range(n_timesteps):
#         if np.argmax(x["type"][timestep_index])==0:
#             note_sequence.append({"pitch":np.argmax(x["pitch"][timestep_index]),
#                                 "start":np.argmax(x["onset"][timestep_index]),
#                                 "end":np.argmax(x["onset"][timestep_index])+np.argmax(x["duration"][timestep_index])})
#     return note_sequence
                
# def noteseq_to_model_format(note_sequence, n_pitches,n_timesteps, sequence_length):
#     """
#     Converts a note sequence to a model input format.
#     ATTENTION: Only takes every other note!!
#     """
#     note_type=[]
#     note_pitch=[]
#     note_onset=[]
#     note_duration=[]
#     for step_index in range(sequence_length):
#         if step_index < len(note_sequence) and note_sequence[step_index]["start"]%2==0:
#             # type section
#             note_type.append(onehot(0,2))
#             # pitch section
#             note_pitch.append(onehot(note_sequence[step_index]["pitch"],n_pitches))
#             # onset section
#             note_onset.append(onehot(note_sequence[step_index]["start"],n_timesteps))
#             # duration section
#             duration = note_sequence[step_index]["end"]-note_sequence[step_index]["start"]
#             note_duration.append(onehot(duration, n_timesteps))
#         else:
#             note_type.append(onehot(1,2))
#             # uniform distribution
#             note_pitch.append(np.ones(n_pitches)/n_pitches)
#             note_onset.append(np.ones(n_timesteps)/n_timesteps)
#             note_duration.append(np.ones(n_timesteps)/n_timesteps)
#     seq = OrderedDict()
#     seq["type"] = np.stack(note_type)
#     seq["pitch"] = np.stack(note_pitch)
#     seq["onset"] = np.stack(note_onset)
#     seq["duration"] = np.stack(note_duration)
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