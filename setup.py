"""
"""

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    entry_points={
        'console_scripts': [
            'run=pet_ct.run:run',
            'create=pet_ct.run:create',
            'connect=pet_ct.run:connect'
        ]
    },
    name="fdg-pet-ct",
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
        'torch', 'torchvision', 'h5py', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels',
        'opencv-python', 'pydicom', 'tqdm', 'Pillow', 'click', 'matplotlib', 'networkx', 'jsmin',
        'ipywidgets', 'nltk', 'sentencepiece', 'plotly', 'tensorboardX', 'pytorch-pretrained-bert',
        'snorkel-metal', 'py-rouge', 'seaborn', 'colorlover'
    ]
)
