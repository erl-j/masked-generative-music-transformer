
#%%
from midi_dataset import MidiDataset
import torch
import torch
import numpy as np

#%%
ds = MidiDataset(prepared_data_path="data/prepared_gamer_data.pt", crop_size=32,downsample_factor=2)

rolls_ = [ds[i]["piano_roll"] for i in range(len(ds))]
#%%
rolls = np.stack(rolls_)

rolls=( rolls>0).astype(np.int32)

three_notes = rolls+np.roll(rolls,1,axis=1)+np.roll(rolls,-1,axis=1)

print(np.max(three_notes))

for i in range(3):
    plt.imshow(three_notes[i])
    plt.show()

has_three_notes = (three_notes>1).any(-1).any(-1)

rolls = rolls[~has_three_notes,:,:]

print(has_three_notes)

roll_sort =  np.argsort(-rolls.sum(1).sum(1))

for roll in roll_sort[:10]:
    plt.imshow(rolls[roll],vmin=0,vmax=1)
    plt.show()

#%% get onsets
onsets = np.diff(rolls,prepend=0,axis=-1)

idx = np.random.randint(0,rolls.shape[0])
plt.imshow(rolls[idx])
plt.show()
plt.imshow(onsets[idx])
plt.show()
plt.imshow(onsets[idx]>0)
plt.show()

# filter away rolls with less than 8 onsets

# filter away rolls with directly neighboring notes being played simultaneously

# filter away non unique rolls.

#%%

#%%
import matplotlib.pyplot as plt
plt.imshow(rolls.sum(0))
plt.show()

plt.plot(rolls.sum(0).sum(0))
plt.show()

plt.plot(rolls.sum(0).sum(1))
plt.show()
#%%
import glob

d = glob.glob("data/GAMER/**/*.mid",recursive=True)

print(len(d))

print(rolls.shape)

#%%
roll_sums = (rolls>0).sum(-1).sum(-1)

print(roll_sums)

plt.hist(roll_sums, bins=100, range=(0,400))
plt.show()

#%%

#%%
# filter away all rolls where more than two notes next to each other are played at once


#dl = torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True)

# for batch in dl:
#     print(batch["piano_roll"].shape)
#     assert batch["piano_roll"].shape[-1] == 32
# for batch in dl:
#     print(batch["piano_roll"].shape)
#     assert batch["piano_roll"].shape[-1] == 32

# %%
