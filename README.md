# Multi-task weak supervision enables automated abnormality localization in whole-body FDG-PET/CT
*Sabri Eyuboglu\*, Geoffrey Angus\*, Bhavik N. Patel, Anuj Pareek, Guido Davidzon, Jared Dunnmon\*\*, Matthew P. Lungren\*\**

This repository includes a PyTorch implementation of our multi-task, weak supervision framework for abnormality localization in large, volumetric medical images.

The experiments use a dataset from the Stanford Hospital of FDG-PET/CT scans and their associated reports. A tutorial on how to load a model in for predictions on custom input can be found at `pet_ct/notebooks/tutorial/notebook.ipynb`.


<p align="center">
<img src="https://github.com/seyuboglu/weakly-supervised-petct/raw/master/data/images/fig1.png" width="800" align="center">
</p>

## Requirements

- Click 7.0
- h5py 2.9.0
- nltk 3.4.4
- numpy 1.16.4
- opencv-python 4.1.0.25
- pandas 0.24.2
- Pillow 6.1.0
- pydicom 1.2.2
- pytorch-pretrained-bert 0.6.2
- pytorch-transformers 1.2.0
- regex 2019.6.8
- rouge 0.3.2
- scikit-learn 0.21.2
- scipy 1.3.0
- seaborn 0.9.0
- sentencepiece 0.1.82
- six 1.12.0
- tensorboardX 1.8
- torch 1.1.0
- torchvision 0.3.0
- tqdm 4.32.2
