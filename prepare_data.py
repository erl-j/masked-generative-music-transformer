#%%
from midi_dataset import MidiDataset
from joblib import Parallel, delayed
import os
#%%

os.makedirs("data/vast",exist_ok=True)
def create_dataset(i):
    ds = MidiDataset(midi_filepaths=f".../{i}/**/*.mid", only_88_keys=False)
    ds.save_data(f"data/vast/prepared_vast_data_{i}.pt")

ROOT_PATH = "data/vast"
source_dirs = os.listdir(ROOT_PATH)

Parallel(n_jobs=4)(delayed(create_dataset)(i) for i in source_dirs)
#%%
