# BTIS-Net
Brain tumor image segmentation based on BTIS-Net
# BTIS-Net for Brain Tumor Image Segmentation

This repository contains the implementation of a BTIS-Net model for brain tumor image segmentation using the BraTS2021 dataset.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project uses a BTIS-Net model for segmenting brain tumors in MRI images, based on the BraTS2021 dataset.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/liulijiangnan/BTIS-Net-Brain-Tumor-Segmentation.git
cd BTIS-Net-Brain-Tumor-Segmentation
pip install -r requirements.txt
Dataset
Download the BraTS2021 dataset from the BraTS2021 website and organize it as follows:
data/
    BraTS2021_00495/
        BraTS2021_00495_t1.nii.gz
        BraTS2021_00495_t1ce.nii.gz
        BraTS2021_00495_t2.nii.gz
        BraTS2021_00495_flair.nii.gz
        BraTS2021_00495_seg.nii.gz
    ...

This repository contains the implementation of a BTIS-Net model for brain tumor image segmentation using the BraTS2021 dataset.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project uses a BTIS-Net model for segmenting brain tumors in MRI images, based on the BraTS2021 dataset.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ll/BTIS-Net-Brain-Tumor-Segmentation.git
cd BTIS-Net-Brain-Tumor-Segmentation
pip install -r requirements.txt
Dataset
Download the BraTS2021 dataset from the BraTS2021 website and organize it as follows:

data/
    BraTS2021_00495/
        BraTS2021_00495_t1.nii.gz
        BraTS2021_00495_t1ce.nii.gz
        BraTS2021_00495_t2.nii.gz
        BraTS2021_00495_flair.nii.gz
        BraTS2021_00495_seg.nii.gz
    ...
Code Structure
dataset.py: Contains the BRATSDataset class for loading and preprocessing the BraTS2021 dataset.
main.py: Main entry point of the project. Handles data loading, model training, evaluation, and inference.
model.py: Contains the implementation of the BTIS-Net model and related components.
train.py: Contains the training loop and logic for training the model.
utils.py: Contains utility functions for data handling, visualization, and other helper functions.
Usage
Training
To train the BTIS-Net model, run:
python train.py --data_dir path/to/BraTS2021_Training_Data --batch_size 8 --epochs 100
Evaluation
To evaluate the model, run:
python evaluate.py --data_dir path/to/BraTS2021_Validation_Data --model_path path/to/saved_model.pth
Inference
To perform inference on new data, run:
python inference.py --data_dir path/to/new_data --model_path path/to/saved_model.pth
Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.


