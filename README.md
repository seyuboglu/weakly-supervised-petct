# Multi-task weak supervision enables automated abnormality localization in whole-body FDG-PET/CT
*Sabri Eyuboglu\*, Geoffrey Angus\*, Bhavik N. Patel, Anuj Pareek, Guido Davidzon, Jared Dunnmon\*\*, Matthew P. Lungren\*\**

This repository includes a PyTorch implementation of a multi-task, weak supervision framework for abnormality localization in large, volumetric medical images, as described in [our manuscript](https://cs.stanford.edu/people/sabrieyuboglu/petct.pdf). Unlike existing weak supervision approaches that use programmatic labeling functions, this one is based on expressive, pre-trained language models that can learn to extract complex labels from unstructured text with just a tiny sample of hand-annotated data. Using these labels, multi-task convolutional neural networks can be trained to  localize features of interest in large volumetric images.

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

[1] Clone the repository
```bash
git clone https://github.com/seyuboglu/weakly-supervised-petct.git
cd weakly-supervised-petct
```
[2] Create a virtual environment and activate it (or activate an existing one)
```
python3.7 -m venv ./env
source env/bin/activate
```

[3] Install the package (`-e` for development mode):
```
pip install -e .
```

## Requirements
- This package has been tested on the following systems: macOS Catalina (10.15) and Ubuntu Linux (16.04.5).
- This package requires Python >=3.7.3
- This package depends on a number of open-source Python packages which can be found in the [setup.py](https://github.com/seyuboglu/weakly-supervised-petct/blob/master/setup.py).
- Training models in this package intractable without GPU compute. We have tested this package on Nvidia TITAN Xp GPUs with Driver Version: 430.50 and CUDA Version: 10.1 


### Data 
The experiments use a dataset from the Stanford Hospital of FDG-PET/CT scans and their associated reports. A tutorial on how to load a model in for predictions on custom input can be found at `pet_ct/notebooks/tutorial/notebook.ipynb`.
### Tutorials

### Experiments