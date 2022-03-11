
# Contrastive Representation Learning for Gaze Estimation

This repository is the official implementation of [GazeCLR](). 

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
python main.py --config_json configs/gazeclr.json --save_path <path/to/save> --is_load_label --same_person --transforms 
```

GazeCLR (Inv+Equiv)

```
python main.py --config_json configs/gazeclr_inv_equiv.json --save_path <path/to/save> --is_load_label --same_person --transforms 
```

## Pre-trained Models

You can download pretrained models here:

- [GazeCLR (Equiv)](https://drive.google.com/file/d/10K_AwVH6H_0P77lR0XHl3iDsfiep2YTP/view?usp=sharing)
- [GazeCLR (Inv+Equiv)](https://drive.google.com/file/d/1dx7ZLd0y-EzWW3wUiQC6BFZvCjF82wf8/view?usp=sharing)

## Questions?

For any inquiries, please contact us.
