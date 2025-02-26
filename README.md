## Does Generation Require Memorization? Creative Diffusion Models using Ambient Diffusion

This repository hosts the official PyTorch implementation of the paper.

## Installation
The recommended way to run the code is with an Anaconda/Miniconda environment.

Create a new Anaconda environment and install the dependencies using the following command: 

`conda env create -f environment.yml -n diffusion`

### Download datasets

You might also need to download dataset and dataset statistics for training and for FID calculation.
To do so, follow the instructions provided [here](https://github.com/NVlabs/edm#preparing-datasets).

To create a small dataset with 300, 1000 and 3000 images, we randomly choose the images from the complete dataset. 

## Training New Models

To train a new model, set the arguments in `scripts/train.sh` and run the script: 

`bash scripts/train.sh`

## Generating images from a model and calculating FID

To generate images from a model, specify the model path `CKPT`, output directory to store the generated images `OUTDIR` and a path to reference statistics `REF_PATH` in `scripts/generate_script.sh` and then run the script: 

`bash scripts/generate_script.sh`

## Calculating similarity scores to measure memorization

To calculate the similarity score, use `scripts/memorization_metrics.sh`. First, specify the path of of generated images in `GEN_DATA` and training data in `DATA` and then run the script: 

`bash scripts/memorization_metrics.sh`

## Acknowledgement

This code is adapted from the [EDM repository](https://github.com/NVlabs/edm).

