"""
"""

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    entry_points={
        'console_scripts': [
            'run=pet_ct.run:run',
            'create=pet_ct.run:create'
        ]
    },
    name="pet-ct",
    version="0.0.1",
    author="Geoff Angus and Sabri Eyuboglu",
    author_email="eyuboglu@stanford.edu",
    description="Research software for AIMI fdg-pet-ct project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seyuboglu/fdg-pet-ct",
    packages=setuptools.find_packages(include=['pet_ct']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'Click==7.0', 
        'h5py==2.9.0', 
        'ipykernel==5.1.1',
        'ipython==7.6.1',
        'ipywidgets==7.5.0',
        'ipyvolume==0.5.2',
        'jupyter==1.0.0',
        'jupyterlab==1.0.2',
        'lifelines==0.23.0',
        'matplotlib==3.1.1',
        'networkx==2.3',
        'nltk==3.4.5',
        'nodejs==0.1.1',
        'numpy==1.16.4',
        'opencv-python==4.1.0.25',
        'pandas==0.24.2',
        'Pillow==8.2.0',
        'pydicom==1.3.0',
        'transformers==2.6.0',
        'scikit-learn==0.21.2',
        'scipy==1.3.0',
        'seaborn==0.9.0',
        'sentencepiece==0.1.82',
        'six==1.12.0',
        'snorkel-metal==0.5.0',
        'tensorboardX==1.8',
        'torch==1.1.0',
        'torchvision==0.3.0',
        'tqdm==4.32.2',
    ]
)
