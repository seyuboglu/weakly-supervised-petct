"""
Classes and functions supporting our HDF5 Dataset.
"""
import logging
import os
from glob import iglob, glob

import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
from PIL import Image


import pet_ct.data.dicom as dicom
from pet_ct.util.util import set_logger, Process, ensure_dir_exists, dir_has_subdirs


class H5Dataset():

    def __init__(self, dataset_name, data_dir, mode="read"):
        """
        args:
            data_dir    (str) directory where the data lives should have
                        dataset, exams, and reports subdirectories.
            dataset_name    (str) the name of the dataset
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.initialize_hdf5(mode)

    def __del__(self):
        """
        Closes the hdf5 file
        """
        self.file.close()

    def initialize_hdf5(self, mode="read"):
        """
        If an HDF5 file does not exist, creates a new one and initializes
        base groups ["/exams/"]. If it does exist, it is simply opened
        for reading and writing. The directory of the dataset is assumed
        to be "{data_dir}/datasets/{dataset_name}.
        args:
            mode    (str) read or write
        """
        dataset_dir = os.path.join(self.data_dir, "datasets")
        ensure_dir_exists(dataset_dir)
        dataset_path = os.path.join(dataset_dir,
                                    "{}.hdf5".format(self.dataset_name))
        exists = os.path.exists(dataset_path)
        if mode == 'read':
            try:
                self.file = h5py.File(dataset_path, 'r')
            except:
                raise(Exception("HDF5 file ({}) must exist when opening in read mode."
                                .format(dataset_path)))
            #logging.info(("Opening HDF5 file for reading: {}").format(self.dataset_name))
        elif mode == "write":
            if not exists:
                logging.info("Creating new HDF5 file: {}".format(self.dataset_name))
            logging.info(("Opening HDF5 file for reading " +
                          "and writing:").format(self.dataset_name))
            self.file = h5py.File(dataset_path, 'a')
            if "exams" not in self.file:
                self.file.create_group("exams")
        else:
            raise(Exception("HDF5 mode not recognized."))

        try:
            self.exams = self.file["exams"]
        except:
            raise(Exception("HDF5 file does not have \"exams\" group."))
        self.mode = mode

    def open_exam(self, exam_id):
        """
        Opens an exam, if it does not exist, it is created.
        args:
            exam_id (str)   the exam id (e.g. LDcb231a)
        returns:
            exam    (group) the exam group in the hdf5 file
        """
        if exam_id not in self.exams:
            if self.mode == "write":
                self.exams.create_group(exam_id)
            else:
                raise(Exception("Exam {} does not exist".format(exam_id)))
        return self.exams[exam_id]

    def write_images(self, exam_id, image_type, images, attrs=None, overwrite=False):
        """
        """
        if self.mode != "write":
            raise(Exception("Read mode: cannot write images to HDF5."))
        exam = self.open_exam(exam_id)
        if image_type not in exam or overwrite:
            dataset = exam.create_dataset(image_type, data=images)
            self._update_attrs(dataset, attrs)

    def write_attrs(self, exam_id, metadata, overwrite=False):
        """
        """
        if self.mode != "write":
            raise(Exception("Read mode: cannot write images to HDF5."))
        exam = self.open_exam(exam_id)
        self._update_attrs(exam, metadata)

    def write_report(self, exam_id, report_text, overwrite=False):
        """
        """
        if self.mode != "write":
            raise(Exception("Read mode: cannot write images to HDF5."))
        exam = self.open_exam(exam_id)
        if "report" not in exam:
            dt = h5py.special_dtype(vlen=str)
            exam.create_dataset("report", data=np.string_(report_text), dtype=dt)
        elif overwrite:
            del exam["report"]
            dt = h5py.special_dtype(vlen=str)
            exam.create_dataset("report", data=np.string_(report_text), dtype=dt)

    def read_images(self, exam_id, image_type, sample_start=None, sample_end=None, sample_rate=1):
        """
        """
        if sample_start == None and sample_end == None:
            return self.exams[exam_id][image_type]

        try:
            images = self.exams[exam_id][image_type][sample_start:sample_end:sample_rate]
        except Exception as e:
            raise(Exception("Can't find images of type {} for exam {}. Original error: {}"
                            .format(image_type, exam_id, str(e))))
        return images

    def read_attrs(self, exam_id):
        """
        """
        try:
            attrs = self.exams[exam_id].attrs
        except:
            raise(Exception("Cannot find attrs for exam {}".format(exam_id)))
        return attrs

    def read_reports(self, exam_id):
        """
        """
        #TODO: test this
        try:
            reports = self.exams[exam_id]["report"]
            return reports[()]
        except:
            raise(Exception("Cannot find reports for exam {}.".format(exam_id)))

    def has_exam(self, exam_id):
        """
        """
        return exam_id in self.file["exams"]

    def has_image_type(self, exam_id, image_type):
        """
        """
        if not self.has_exam(exam_id):
            return False
        return image_type in self.file["exams/{}".format(exam_id)]

    def has_reports(self, exam_id):
        """
        """
        if not self.has_exam(exam_id):
            return False
        return "reports" in self.file["exams/{}".format(exam_id)]

    def _update_attrs(self, object, attrs, overwrite=False):
        """
        Updates the attributes for an HDF5 object.
        args:
            object  (group or dataset)  an HDF5 object with attrs (i.e. group/dataset)
            attrs   (dict)  dictionary of attributes to include
            overwrite   (bool)  if true, change value of attrs that already exist
        """
        if attrs is None:
            return
        print(attrs)
        for key, value in attrs.items():
            if key not in object.attrs or overwrite:
                object.attrs[key] = value


class DatasetEditor(Process):

    def __init__(self, dir, dataset_name, data_dir, reports_path=None, attrs_path=None):
        """
        """
        super().__init__(dir)

        self.dataset = H5Dataset(dataset_name, data_dir, mode="write")
        self.reports_path = reports_path
        self.attrs_path = attrs_path


    def _run(self, overwrite=False):
        """
        """
        if self.reports_path is not None:
            logging.info("Editing reports...")
            reports_df = pd.read_csv(self.reports_path, index_col=0)
            for exam_id in tqdm(self.dataset.file["exams"].keys()):
                report_txt = reports_df.loc[exam_id]["report_txt"]
                self.dataset.write_report(exam_id, report_txt, overwrite=True)

        if self.attrs_path is not None:
            logging.info("Editing attributes...")
            reports_df = pd.read_csv(self.attrs_path, index_col=0)
            for exam_id in tqdm(self.dataset.file["exams"].keys()):
                metadata = reports_df.loc[exam_id].to_dict()
                self.dataset.write_attrs(exam_id, metadata, overwrite=False)


class DatasetBuilder(Process):

    def __init__(self, process_dir):
        """
        Inherits from process class which sets all parameters
        as instance variables of the class.
        """
        super().__init__(process_dir, name="builder")
        logging.info("Running HDF5 dataset build process at: {}".format(self.dir))

        self.exams_root_dir = os.path.join(self.data_dir, "exams")
        self.dataset_dir = os.path.join(self.data_dir, "datasets",
                                        self.dataset_name + ".hdf5")
        logging.info("Moving exams from {} to HDF5 file at {}".format(self.exams_root_dir,
                                                                      self.dataset_dir))

        self.reports = load_reports(self.reports_root_dir)

        self.dataset = H5Dataset(self.dataset_name, self.data_dir, mode="write")
        self.exams = {}
        self.stats = {"written": 0,
                      "skipped": 0,
                      "failed": 0}

    def _run(self, overwrite=False):
        """
        """
        self.build(overwrite)

    def build(self, overwrite=False):
        """
        Buildings
        Args:
            root_dir    Location in which we can find <ABNORMALITY_LEVEL> folders.
            db_path     Location in which we will write the SQL database file.
        """
        logging.info("Building dataset...")
        with tqdm(total=self.expected_exams) as t:
            for exam_dir in iglob("{}/*/*/*/".format(self.exams_root_dir)):
                if hasattr(self, "limit") and self.stats["written"] >= self.limit:
                    break
                # don't write exam if it is missing image types
                if not dir_has_subdirs(exam_dir, self.image_types):
                    self.stats["skipped"] += 1
                    logging.info("{} Skipping exam at: {}".format(self._get_completion_str(),
                                                                exam_dir))
                    continue

                # try to write exam, log warning otherwise
                try:
                    self._write_exam(exam_dir, overwrite)
                    self.stats["written"] += 1
                    t.update()
                    logging.info("{} Wrote exam at: {}".format(self._get_completion_str(),
                                                               exam_dir))
                except Exception as e:
                    logging.warning(("{} Failed to write exam at: {} \n" +
                                    "Error Message: {}").format(self._get_completion_str(),
                                                                exam_dir, e))
                    self.stats["failed"] += 1
        self._output()
        logging.info("Done.")

    def _output(self):
        """
        Outputs the exams written to csv. Also outputs statistics from the build to csv.
        """
        self.exams_df = pd.DataFrame.from_dict(self.exams, orient='index')
        self.exams_df.to_csv(os.path.join(self.dir, "exams.csv"))
        self.stats_df = pd.DataFrame.from_dict([self.stats])
        self.stats_df.to_csv(os.path.join(self.dir, "stats.csv"))

    def _write_exam(self, exam_dir, overwrite=False):
        """
        Write one exam to the dataset.
        args:
            exam_dir    (str) the exam's directory
        """
        label, patient_id, exam_id = self._unpack_directory(exam_dir)
        attrs = {"exam_id": exam_id,
                 "patient_id": patient_id,
                 "label": label,
                 "exam_dir": exam_dir
                }

        for image_type in self.image_types:
            # skip adding images if they are already in exam
            if self.dataset.has_image_type(exam_id, image_type) and not overwrite:
                continue
            paths, dcms, imgs = dicom.parse_dicoms_from_dir(os.path.join(exam_dir,
                                                                         image_type))
            # stack the images along a new axis
            imgs = np.stack(imgs, axis=0)

            # add dicom attrs to metadata, take from first dicom
            image_attrs = dicom.extract_attrs(dcms[0], self.image_attrs)
            image_attrs.update({"dim_{}".format(i): d for i, d in enumerate(imgs.shape)})
            attrs.update({"{}/{}".format(image_type, attr): value
                         for attr, value in image_attrs.items()})

            self.dataset.write_images(exam_id, image_type, imgs, image_attrs, overwrite)
        self.dataset.write_report(exam_id, self.reports[exam_id]["report_txt"])
        self.dataset.write_attrs(exam_id, attrs, overwrite)
        self.exams[exam_id] = attrs

    def _resize(self, imgs):
        """
        """
        if not hasattr(self, "shape"):
            return imgs

        return [torch.tensor(Image.fromarray(img).resize(self.shape)) for img in imgs]

    def _get_completion_str(self):
        """
        """
        if hasattr(self, "expected_exams"):
            prcnt = round(100 * self.stats["written"] / self.expected_exams, 2)
            return "[{}/{} : {}%]".format(self.stats["written"],
                                          self.expected_exams,
                                          prcnt)
        else:
            return "[{}/??: ??%]".format(self.stats["written"])

    def _load_reports(self):
        """
        Loads the reports
        """
        self.reports_root_dir = os.path.join(self.data_dir, "reports")
        self.reports_df = pd.read_excel(os.path.join(self.reports_root_dir,
                                                     "reports.xlsx"))
        self.acc_nums_df = pd.read_csv(os.path.join(self.reports_root_dir,
                                                    "acc_nums.csv"),
                                       header=None, names=["acc_nums", "exam_dir"],
                                       index_col=0)

        self.reports = {}
        logging.info("Loading reports...")
        for acc_num, row in tqdm(self.acc_nums_df.iterrows(),
                                 total=self.acc_nums_df.shape[0]):
            label, patient_id, exam_id = self._unpack_directory(row["exam_dir"])

            # get row with corresponding accession number
            report = self.reports_df.loc[self.reports_df['acc_num'] == acc_num]
            if report.shape[0] > 0:
                # takes first report with accession number
                self.reports[exam_id] = {"description": report["description"].iloc[0],
                                         "report_txt": report["report_txt"].iloc[0],
                                         "classification": report["Classification"].iloc[0]
                                        }
        logging.info("Done.")

    def _unpack_directory(self, exam_dir):
        """
        Get the label, patient id, and exam id from an exam directory.
        args:
            exam_dir    (str) the exam directory
        returns:
            label   (str)   One of "1", "2", "4" or "9"
            patient (str)   Patient ID (e.g. LDcb21f2)
            exam    (str)   Exam ID (e.g. LDcb231a)
        """
        label, patient, exam = exam_dir.strip('/').split('/')[-3:]
        return label, patient, exam

