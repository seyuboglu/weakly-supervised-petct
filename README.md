# Multi-task weak supervision enables automated abnormality localization in whole-body FDG-PET/CT
*Sabri Eyuboglu\*, Geoffrey Angus\*, Bhavik N. Patel, Anuj Pareek, Guido Davidzon, Jared Dunnmon\*\*, Matthew P. Lungren\*\**

This repository includes a PyTorch implementation of our multi-task, weak supervision framework for abnormality localization in large, volumetric medical images.

The experiments use a dataset from the Stanford Hospital of FDG-PET/CT scans and their associated reports. A tutorial on how to load a model in for predictions on custom input can be found at `pet_ct/notebooks/tutorial/notebook.ipynb`.


<p align="center">
<img src="https://github.com/seyuboglu/weakly-supervised-petct/raw/master/data/images/fig1.png" width="600" align="center">
</p>

## Installation
Clone the repository

```bash
git clone https://github.com/seyuboglu/weakly-supervised-petct.git
cd weakly-supervised-petct
```

Create a virtual environment and activate it
```
python3.7 -m venv ./env
source env/bin/activate
```

Install package (`-e` for development mode)
```
pip install -e .
```
