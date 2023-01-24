from midi_dataset import MidiDataset
import glob
import torch

dataset = "vast+gamer"

CROP_SIZE = 36
DOWNSAMPLE_FACTOR = 1
MODE= "note_seq"

dss = []
if dataset == "gamer":
    ds = MidiDataset(prepared_data_path="data/prepared_gamer_data.pt",crop_size=CROP_SIZE,downsample_factor=DOWNSAMPLE_FACTOR,mode="note_seq")
    dss.append(ds)
if (dataset == "vast") or (dataset == "vast+gamer"):
    fps = glob.glob("data/vast/*.pt")
    for fp in fps:
        ds = MidiDataset(prepared_data_path=fp,crop_size=CROP_SIZE,downsample_factor=DOWNSAMPLE_FACTOR,mode="note_seq")
        dss.append(ds)

ds = torch.utils.data.ConcatDataset(dss)

# save dataset
torch.save(ds,f"data/prepared_{dataset}_noteseq_data.pt")