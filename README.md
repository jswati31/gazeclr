
# Contrastive Representation Learning for Gaze Estimation

This repository is the official PyTorch implementation of [GazeCLR](https://arxiv.org/abs/2210.13404).

- Published at NeurIPS 2022, Gaze Meets ML (**Best Paper Award**, Spotlight)
- Authors: [Swati Jindal](https://jswati31.github.io/), [Roberto Manduchi](https://users.soe.ucsc.edu/~manduchi/)


## Requirements
The code is tested with Python 3.7.10 and torch 1.18.1.

To install all the packages:

```setup
pip install -r requirements.txt
```


## Data Processing

1. Download [EVE Dataset](https://ait.ethz.ch/projects/2020/EVE/)
2. We extract all frames from videos in pre-processing stage for faster dataloading during training. Run
```
python video2imgs.py
```

## Train GazeCLR

GazeCLR (Equiv)

```
python main.py --config_json configs/gazeclr.json --save_path <path/to/save> --is_load_label --same_person
```

GazeCLR (Inv+Equiv)

```
python main.py --config_json configs/gazeclr_inv_equiv.json --save_path <path/to/save> --is_load_label --same_person
```

## Pre-trained Models

You can download pretrained models here:

- [GazeCLR (Inv+Equiv)](https://drive.google.com/file/d/17_BdB-mnsZEw33yaqHZxJYX8YLArfcq9/view?usp=sharing)
- [GazeCLR (Equiv)](https://drive.google.com/file/d/1qGwQnCLkxHitrj1QLhBGDVLfmvEYU0r1/view?usp=sharing)

## Questions?

For any inquiries, please contact at swjindal@ucsc.edu
