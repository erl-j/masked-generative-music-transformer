#%%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Audio, display

from chiptune import chiptunes_synthesize
from midi_dataset import MidiDataset
from piano_roll_to_pretty_midi import to_pretty_midi
from train import Model
from util import crop_roll_128_to_88, pad_roll_88_to_128


#%%
def play_audio(audio):
    display(Audio(audio, rate=44100))
ds = MidiDataset(prepared_data_path="data/prepared_gamer_data.pt", crop_size=36,downsample_factor=2)

dl = torch.utils.data.DataLoader(ds,batch_size=1,shuffle=True)

#%%
# Hide the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(n_pitches=36,n_timesteps=32, architecture = "transformer", n_layers=5,n_hidden_size=256)
ckpt_path =glob.glob("lightning_logs/2rv228sb/checkpoints/*.ckpt")[0]

#ckpt_path = "artefacts/epoch=14-step=187080.ckpt"
#ckpt_path =glob.glob("lightning_logs/2c1uhoqg/checkpoints/*.ckpt")[0]
# n_layers = 5, n_hidden_size = 256

#%%
# test
model.eval()
for i in range(3):
    x= torch.zeros((1,36,32,2))
    x[0,0,0,0] = 1 

    mask = torch.ones((1,36,32,1))
    mask[0,0,0,0] = 0

    y,y_probs = model.forward(x,mask)
    print(y_probs[0,0,1])

#%%


# Load the model
model.load_state_dict(torch.load(ckpt_path,map_location=torch.device(device))['state_dict'])
# %%
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
idx = np.random.randint(0,len(ds))
roll = torch.tensor(ds[idx]["piano_roll"])

roll =roll
mask = torch.zeros_like(roll)

plt.imshow(roll)
plt.show()

roll = roll[None,...]

play_roll(roll[0])

mask  = mask[None,...]

mask[:,:,10:]=1
roll[:,:,10:]=0

play_roll(roll[0])

mask = mask.float()

roll = model.generate(30,0.1,x=roll, mask=mask,use_confidence_sampling=False,plot=True)[0]

play_roll(roll)

plt.imshow(roll)
plt.show()
#%%
N_PITCHES=36
N_TIMESTEPS=32

roll = np.zeros((1,N_PITCHES,N_TIMESTEPS))
mask = np.zeros((1,N_PITCHES,N_TIMESTEPS))

# major scale
#scale = [0,2,4,5,7,9,11]
# minor scale
# scale = [0,2,3,5,7,8,10]
# major pentatonic
#scale = [0,2,4,7,9]
scale =[0,1,2,3,4,5,6,7,8,9,10,11]
for i in range(1+(N_PITCHES//12)):
    for j in scale:
        note_idx = i*12+j
        if note_idx < N_PITCHES:
            mask[:,note_idx,:]=1

# set a random value to 1
#roll[:,np.random.randint(0,N_PITCHES),np.random.randint(0,N_TIMESTEPS)]=1
roll = torch.tensor(roll).float()
mask  = torch.tensor(mask).float()

roll[0,12,0] = 1
mask[0,12,0] = 0

roll = model.generate(60,1.0,x=roll,mask=mask,activity_bias=0.0,use_confidence_sampling=False,plot=True)[0]
 

#roll = model.generate_with_random_order(1,100)[0]

# idx = np.random.randint(0,len(ds))
# roll = torch.tensor(ds[idx]["piano_roll"])
plt.imshow(roll)
plt.show()

play_roll(roll)
# %%
