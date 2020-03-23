"""
A utilities file containing functions to extract and parse DICOM files
for visualization.
"""
import os
from datetime import datetime
from glob import iglob
from collections import Counter

import pydicom
from pydicom import dcmread
import pandas as pd
from tqdm import tqdm
import numpy as np

from pet_ct.util.util import Process


# Added on 5/3
def get_dicom_paths(exam_dir, image_type):
    dicom_dir = os.path.join(exam_dir, image_type)
    dicom_filenames = os.listdir(dicom_dir)
    dicom_filenames = sorted(dicom_filenames, key=lambda x: x[:6])
    dicom_paths = [os.path.join(dicom_dir, filename) for filename in dicom_filenames]
    return dicom_paths


def get_dicom_images(exam_dir, image_type):
    dicom_paths = get_dicom_paths(exam_dir, image_type)
    imgs = []
    for path in tqdm(dicom_paths):
        dicom = dcmread(path)
        img = dicom.pixel_array
        imgs.append(img)
    return np.array(imgs)


def get_exam_from_source(exam_dir):
    imgs_ct = get_dicom_images(exam_dir, "CT Images")
    imgs_pet = get_dicom_images(exam_dir, "PET_BODY_CTAC")
    return {
        'CT Images': imgs_ct,
        'PET_BODY_CTAC': imgs_pet
    }
# end

class DicomParser(Process):
    """Extracts the attributes specified in each exam's DICOM data.

    The DicomParser class is used in support of an existing dataset. It
    requires an exams.csv file in parent directory in order to determine the
    target exams.

    params:
        process_dir (string) the process directory.
        data_dir (string)    the directory containing all exams.
        exam_type (string)   the directory to search within an exam for DICOMs.
        attr_keys (list)     the attributes of interest in the DICOMs.
    """
    def __init__(
        self, process_dir, data_dir="/data4/data/fdg-pet-ct/exams",
        exam_type="PET_BODY_CTAC", attr_keys=[],
    ):
        self.dir = process_dir
        self.data_dir = data_dir
        self.exam_type = exam_type
        self.attr_keys = attr_keys

        self.exams_path = os.path.join(self.dir, "..", "exams.csv")
        if not os.path.exists(self.exams_path):
            raise(Exception("Parent directory of dicom_attrs directory must contain an \
                             an \"exams.csv\" file."))
        self.exams_df = pd.read_csv(self.exams_path)

    def _run(self, overwrite=False):
        self._extract(overwrite)

    def _extract(self, overwrite):
        """Extracts the attributes and places them in a dictionary of lists.

        Leverages the very particular folder structure inherent to our dataset,
        which takes the following form:

        <data_dir>/<exam_label>/<patient_id>/<exam_id>
        """
        out_path = os.path.join(self.dir, 'dicom_attrs.csv')
        if os.path.exists(out_path) and not overwrite:
            raise(Exception(f"File dicom_attrs.csv already exists in \
            directory {self.dir}. Use --overwrite flag to overwrite."))

        indexes = []
        entries = []
        for idx, row in tqdm(self.exams_df.iterrows(), total=self.exams_df.shape[0]):
            exam_label = str(row['label'])
            patient_id = row['patient_id']
            exam_id = row['exam_id']
            exam_dir = os.path.join(
                self.data_dir, exam_label, patient_id, exam_id, self.exam_type
            )
            dcm_path = next(iglob(os.path.join(exam_dir, '*.dcm')))
            dcm = pydicom.dcmread(dcm_path)
            dcm_dict = dicom_dictify(dcm)
            indexes.append(exam_id)
            entries.append({k: v for k, v in dcm_dict.items() if k in self.attr_keys})

        attrs_df = pd.DataFrame(data=entries, index=indexes)
        attrs_df.to_csv(out_path)


def dicom_dictify(ds):
    """Turn a pydicom Dataset into a dict with keys derived from the Element tags.

    source: https://github.com/pydicom/pydicom/issues/319

    Parameters
    ----------
    ds : pydicom.dataset.Dataset
        The Dataset to dictify

    Returns
    -------
    output : dict
    """
    output = dict()
    for elem in ds:
        if type(elem.value) == bytes:
            continue
        if elem.VR != 'SQ':
            output[f'{str(elem.tag)} {elem.name}'] = elem.value
        else:
            output[f'{str(elem.tag)} {elem.name}'] = [dicom_dictify(item)
                                                      for item in elem]
    return output


def parse_dicom(path):
    """A function to convert a DICOM file into pythonic structures.

    The caller must pass in the absolute path for simplicity's sake.

    Arguments:
        dirname     The ABSOLUTE path to the DICOM directory.
        filename    The name of a selected DICOM file.

    Returns:
        A tuple containing the following:
        - A Pydicom Dataset object
        - A numpy array containing the DICOM image itself.
    """
    dcm = pydicom.dcmread(os.path.join(path))
    img = dcm.pixel_array
    return dcm, img

def parse_dicoms_from_dir(root_dirname, img_only=False, delete_dups=True):
    """Returns a list of Dataset objects given a directory of dicoms.

    First, all DICOM filenames are extracted. These are then iterated
    through. More on Pydicom Dataset objects here:
    https://pydicom.github.io/pydicom/stable/api_ref.html

    Arguments:
        root_dirname    The ABSOLUTE path to the DICOM root directory.
        img_only        Save images only, reducing memory requirement.

    Returns:
        A tuple containing the following:
        - A list of Pydicom Dataset objects
        - A list of tuples containing (dir, filename) for each DICOM
        - A list of numpy arrays containing the DICOM images themselves
    """
    dcm_paths = []
    for dirname, subdirnames, filenames in os.walk(root_dirname):
        for filename in filenames:
            if ".dcm" in filename.lower():
                dcm_paths.append((dirname, filename))

    dcms = []
    imgs = []

    for dirname, filename in dcm_paths:
        dcm, img = parse_dicom(os.path.join(dirname, filename))
        imgs.append(img)

        if not img_only:
            dcms.append(dcm)

    dcm_img_tuples = zip(dcm_paths, dcms, imgs)
    dcm_img_tuples = sorted(dcm_img_tuples, key=lambda x: x[1].ImagePositionPatient[2])

    # assumption: full body scans always start LOWER than brain scans and so will be first in the queue.
    # cleans up duplicate (possibly isolated brain) scans
    if delete_dups:
        res_tuples = []
        prev_filename = None
        for curr_path, dcm, img in dcm_img_tuples:
            curr_filename = curr_path[1]

            if prev_filename == None:
                prev_filename = curr_filename
                res_tuples.append((curr_path, dcm, img))
                continue

            prev_dcm_idx = int(prev_filename[:6]) # I believe all DICOMS start with a 6-digit sequence number.
            curr_dcm_idx = int(curr_filename[:6]) # I could be wrong.
            # dcm idx are either toe up or head down.
            if not(prev_dcm_idx + 1 != curr_dcm_idx and prev_dcm_idx - 1 != curr_dcm_idx):
                prev_filename = curr_filename
                res_tuples.append((curr_path, dcm, img))

        paths, dcms, imgs = zip(*res_tuples)
    else:
        paths, dcms, imgs = zip(*dcm_img_tuples)

    return paths, dcms, imgs
