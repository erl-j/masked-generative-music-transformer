#%%
import glob
import torch
from train import Model
import torch.onnx
import os
import onnx
import onnx
import onnxruntime as ort
import numpy as np


model = Model(n_pitches=36,n_timesteps=32, architecture = "transformer",n_layers=4,n_hidden_size=512)
ckpt_path =glob.glob("lightning_logs/2mov0b71/checkpoints/*.ckpt")[0] 
name = "guillaume"

# model = Model(n_pitches=36,n_timesteps=32, architecture = "transformer",n_layers=3,n_hidden_size=128)
# ckpt_path =glob.glob("lightning_logs/1cubm9kd/checkpoints/*.ckpt")[0]
# name="tiny"

# model = Model(n_pitches=36,n_timesteps=32, architecture = "transformer",n_layers=5,n_hidden_size=256)
# ckpt_path =glob.glob("lightning_logs/2rv228sb/checkpoints/*.ckpt")[0]
# name = "model"

print("Loading model from",ckpt_path)
model.load_state_dict(torch.load(ckpt_path)['state_dict'])
example_inputs=(torch.zeros(1,36,32,2),torch.zeros(1,36,32,1))

model.eval()

print("Testing model")
y,yp=model.forward(example_inputs[0],example_inputs[1])

print(model.__dict__)

print("Exporting model to ONNX")
torch.onnx.export(model,               # model being run
                  example_inputs,                         # model input (or a tuple for multiple inputs)
                  f"artefacts/{name}.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['x','mask'],   # the model's input names
                  output_names = ['y','y_probs'], # the model's output names
)

print("Checking model")
onnx_model = onnx.load(f"artefacts/{name}.onnx")
onnx.checker.check_model(onnx_model)

print("Running model with ONNX Runtime")
x = example_inputs[0].numpy()
mask = example_inputs[1].numpy()
ort_sess = ort.InferenceSession(f'artefacts/{name}.onnx')
y2,yprob2 = ort_sess.run(None, {'x': x, 'mask': mask})

assert np.allclose(y2,y.detach().numpy())
assert np.allclose(yprob2,yp.detach().numpy())