# Handwriting | Predict & Synsthesis

Proposed implementation of Alex Grave [paper](https://arxiv.org/abs/1308.0850)

## Results

### Handwritten synthesis of a dactilographed text  
text : Welcome to Lyrebird
![Example synthesis](/data/readme/synsthesis.jpeg)

Attention windows
![Attention window](/data/readme/attention.jpg)

### Random handwriting generation

![Example synthesis](/data/readme/predict.jpeg)

## How to use this repository

This repository is divided into 4 folders:
* data
* notebooks
* models
* utils

### Notebooks folder

* Notebook to explore the data
* Notebook for training models
* Notebook to view the results

### Models folder

* the __dummy__ file gathers the functions able to generate the strokes (unconditional and conditional) from the models learned and stored in ../data/files
* the __dataloader__ file contains the two classes allowing to manipulate easily the dataset with pytorch
* the __networks__ file contains the two classes modeling the two networks for the first two first questions
* the __trainers__ file contains the classes that allow to train the networks preceding
* the __files__ folder contains the saved weights of networks and others 

### Utils folder
* contains the function plot_stroke in the __\_init\___
* contains the __Onehot Encoder__ class which allows to work with Onehot encoded sentences.

### Data folder
* contains the raw data :
    * __strokes-py3.npy__ contains 6000 sequences of points that correspond to handwritten sentences
    * __sentences.txt__ contains the corresponding text sentences


# Requirements

* numpy==1.17.4
* matplotlib==3.1.1
* torch==1.3.1

