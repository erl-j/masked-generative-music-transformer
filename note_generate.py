#%%
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Audio, display

from chiptune import chiptunes_synthesize
from note_dataset import (NoteDataset, model_format_to_noteseq,
                          noteseq_to_model_format)
from piano_roll_to_pretty_midi import to_pretty_midi
from note_train import Model
from util import crop_roll_128_to_88, noteseq_to_roll, pad_roll_88_to_128


def play_audio(audio):
    display(Audio(audio, rate=44100))

def play_roll(roll):
    # pad to 88 with zeros
    roll = torch.nn.functional.pad(roll,(0,0,30,22))
    full_roll = pad_roll_88_to_128(roll)
    # repeat full roll twice
    full_roll = torch.cat([full_roll,full_roll],dim=-1)
    full_roll=full_roll.numpy()
    full_roll=full_roll.T
    midi = to_pretty_midi(full_roll, constant_tempo=100)
    audio = chiptunes_synthesize(midi)
    play_audio(audio)

#%%
ds = NoteDataset(prepared_data_path="data/prepared_gamer_noteseq_data_1000.pt", crop_size=36)

dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

batch = next(iter(dl))["seq"]

sample = {k:v[0] for k,v in batch.items()}

ns = model_format_to_noteseq(sample)


ns = [{**n,"velocity":127} for n in ns]
roll = noteseq_to_roll(ns,36,64)

plt.imshow(roll)
plt.show()

#%%
#%%
# Hide the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(token_sections = ds.get_token_sections(), n_layers=4,n_hidden_size=512)
ckpt_path =glob.glob("lightning_logs/1msm59m1/checkpoints/*.ckpt")[0]

model.load_state_dict(torch.load(ckpt_path,map_location=torch.device(device))['state_dict'])
# %%

x = batch
n_timesteps = x["type"].shape[1]
n_sections = len(x)

n_notes=40
x["type"]=torch.zeros_like(torch.tensor(x["type"]))
x["type"][:,:,0]=0
x["type"][:,:,1]=1
x["type"][:,:n_notes,0]=1
x["type"][:,:n_notes,1]=0

section_mask = torch.ones((1,n_timesteps,n_sections))
section_mask[:,:,0]=0

x = model.generate(x=x,section_mask=section_mask,temperature=1.5, mode="channel")


x0 = {key: tensor[0] for key, tensor in x.items()}
ns = model_format_to_noteseq(x0)
ns = [{**n,"velocity":127} for n in ns]
roll = noteseq_to_roll(ns,36,64)

plt.imshow(roll)
plt.show()

print(len(ns))

play_roll(torch.tensor(roll))

# %%
