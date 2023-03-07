# Masked Generative Music Transformer

[![DOI](https://zenodo.org/badge/591120173.svg)](https://zenodo.org/badge/latestdoi/591120173)

This project applies techniques from MaskGIT [1] / Muse [2] to symbolic music.

![](misc/gen.gif)


We currently use simple piano rolls of size 36 pitches X 32 timesteps as our target representation.

The goal of the project is to apply a pretrained model to tasks such as:
- Scale/Rhythm constrained generation.
- "Genre" conditioning.
- Velocity generation..
- Microtiming generation etc..

We are also working on applying the model to other music representations.

[1]: https://arxiv.org/abs/2202.04200
[2]: https://arxiv.org/abs/2301.00704

If you found this project useful for you research, please cite:


```BibTex
@software{Jonason_Masked_Generative_Music_2023,
author = {Jonason, Nicolas},
doi = {10.5281/zenodo.7703863},
month = {3},
title = {{Masked Generative Music Transformer}},
version = {first},
year = {2023}
}
```
