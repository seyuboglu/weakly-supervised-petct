# Multi-task weak supervision enables automated abnormality localization in whole-body FDG-PET/CT

This code implements the abnormality localization algorithm from the following paper:

> Sabri Eyuboglu*, Geoffrey Angus*, Bhavik Patel, Anuj Pareek, Guido Davidzon, Jared Dunnmon**, and Matthew P. Lungren**
>
> [Multi-task weak supervision enables automated abnormality localization in whole-body FDG-PET/CT]

The experiments use a dataset from the Stanford Hospital of FDG-PET/CT scans and their associated reports. A tutorial on how to load a model in for predictions on custom input can be found at `pet_ct/notebooks/tutorial/notebook.ipynb`.

## Abstract

Recent advances in machine learning for medical imaging have been fueled by large collections of carefully annotated images. However, for whole-body FDG-PET/CT, clinical interpretation often requires localizing pathology to specific organs and regions, and neither fine-grained, anatomical labels nor model architectures designed to use them are generally available. Further, the sheer size and complexity of each scan make retrospective data annotation extremely expensive.
To address these challenges, we leverage recent advancements in natural language processing to develop a weak supervision framework that extracts imperfect, yet highly granular regional abnormality labels from the radiologist reports associated with each scan. Our approach automatically labels each region in a custom ontology of 96 anatomical zones, providing a structured profile of the pathology in each scan. For the scans themselves, we design an attention-based, multi-task 3D-CNN architecture that can be trained using these generated labels to detect abnormalities in the 26 anatomical regions most commonly of interest in FDG-PET/CT protocols. This model provides clinical value in several ways: automatically localizing abnormalities, screening negative exams, identifying rare pathologies, and forecasting patient mortality. We show that our model can detect and localize lymph node, lung and liver abnormalities -- the three regions with highest pathology incidence in our dataset -- with median areas under the ROC curve (AUROC) of 87\%, 85\%, and 92\% respectively, enabling high-sensitivity negative screening and automated triage. We further show that the representation learned during this procedure can be used to train a model that predicts mortality within 90 days at an AUROC of 80\%, which would provide critical information for palliative care teams. In addition to evaluating absolute model performance, we demonstrate via a series of ablation studies that our multi-task formulation is important for achieving results that have clinical value. In regions such as the adrenal gland and pancreas for which positive examples are rare, we observe that multi-task pretraining enables improvements of up to 20 points AUROC over a strong single-task baseline while reducing computational cost by over 50\%. Further, our attention-based CNN improves region-specific performance by up to 20 points AUROC. In summary, our work demonstrates that using expressive language models for multi-task, cross-modal weak supervision of appropriately specified CNN models enables anatomically resolved abnormality detection and mortality prediction in FDG-PET/CT scans while requiring almost no hand-labeled data.  

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
