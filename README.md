# Multi-task weak supervision enables anatomically-resolved abnormality detection in whole-body FDG-PET/CT
*Sabri Eyuboglu\*, Geoffrey Angus\*, Bhavik N. Patel, Anuj Pareek, Guido Davidzon, Jared Dunnmon\*\*, Matthew P. Lungren\*\**

This repository includes a PyTorch implementation of a multi-task, weak supervision framework for abnormality localization in large, volumetric medical images, as described in [our manuscript](https://github.com/seyuboglu/weakly-supervised-petct/raw/master/manuscript.pdf). Unlike existing weak supervision approaches that use programmatic labeling functions, this one is based on expressive, pre-trained language models that can learn to extract complex labels from unstructured text with just a tiny sample of hand-annotated data. Using these labels, multi-task convolutional neural networks can be trained to  localize features of interest in large volumetric images.

 | Section | Description |
|-|-|
| [Installation](#installation) | How to install the package |
| [Tutorials](#tutorials) | Jupyter Notebook tutorials for training label models, scan models,  |
| [Package](#tutorials) | Overview of core modules and classes within the package |
| [Experiments](#experiments) | How to replicate experiments and analyses from our manuscript|
| [Data](#data) | How to use your own data|

<p align="center">
<img src="https://github.com/seyuboglu/weakly-supervised-petct/raw/master/data/images/fig1.png" width="600" align="center">
</p>

## Installation

[1] Clone the repository [~1 min]
```bash
git clone https://github.com/seyuboglu/weakly-supervised-petct.git
cd weakly-supervised-petct
```
[2] Create a virtual environment and activate it (or activate an existing one) [~1 min]
```
python3.7 -m venv ./env
source env/bin/activate
```

[3] Install the package and dependencies (`-e` for development mode) [~5 min]:
```
pip install -e .
```

### Requirements
- This package has been tested on the following systems: macOS Catalina (10.15) and Ubuntu Linux (16.04.5).
- This package requires Python >=3.7.3
- This package depends on a number of open-source Python packages which can be found in the [setup.py](https://github.com/seyuboglu/weakly-supervised-petct/blob/master/setup.py).
- Training models in this package intractable without GPU compute. We have tested this package on Nvidia TITAN Xp GPUs with Driver Version: 430.50 and CUDA Version: 10.1 

## Tutorials
### Preparing Data
`tutorials/01_data/data_tutor.ipynb`  
How to prepare data for our multi-task, weak supervision framework.  We use  [HDF5](https://www.hdfgroup.org/solutions/hdf5/), a file format optimized for high-dimensional heterogeneous datasets (like ours!). Using HDF5 allows us to store all of the data (i.e. scan, report, and metadata) for each exam in our dataset in one place. Unfortunately, many of you are probably unfamiliar with the HDF5 interface, so below we walk you through how to prepare an HDF5 dataset with volumetric imaging data for use in our framework!

In this notebook we:  

1. Go through the motions of preparing an HDF5 dataset for use with our framework. We prepare the dataset with dummy data, but show how you can replace a few functions with custom ones for loading your own data. 

2. Show how we can use this HDF5 dataset with the PyTorch `Dataset` classes we've implemented such as `pet_ct.learn.datasets.MTClassifierDataset`. (TODO)

### Labeling a Dataset for Abnormality Localization
`tutorials/02_labeling/labeling_tutor.ipynb`  
How to perform inference on the task of abnormality localization with a pretrained scan model.  
TODO

### Abnormality Localization Training
`tutorials/03_training/training_tutor.ipynb`  
How to perform inference on the task of abnormality localization with a pretrained scan model.  
In this notebook we cover: 
1. Loading model configurations from a JSON like the one at `tutorials/inference/params.json`
2. Building a `pet_ct.model.MTClassifierModel` and loading pretrained weights (Note: we do not provide pretrained weights for our models to protect PHI.)
3. How input to the model should be structured
4. How to perform inference on the model using `pet_ct.model.MTClassifierModel.predict`
5. How output is structured

### Abnormality Localization Inference
`tutorials/04_inference/inference_tutor.ipynb`  
How to perform inference on the task of abnormality localization with a pretrained scan model.  
In this notebook we cover: 
1. Loading model configurations from a JSON like the one at `tutorials/inference/params.json`
2. Building a `pet_ct.model.MTClassifierModel` and loading pretrained weights (Note: we do not provide pretrained weights for our models to protect PHI.)
3. How input to the model should be structured
4. How to perform inference on the model using `pet_ct.model.MTClassifierModel.score`
5. How output is structured

### Experiments
The parameters and results for all the experiments reported in our manuscript can be found in the `experiments` directory. 

### Data 
The experiments use a dataset from the Stanford Hospital of FDG-PET/CT scans and their associated reports. A tutorial on how to load a model in for predictions on custom input can be found at `pet_ct/notebooks/tutorial/notebook.ipynb`.

