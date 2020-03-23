"""
Provides functions for processing reports.
"""

import os

import pandas as pd
from tqdm import tqdm

from pet_ct.util.util import Process


class ReportsBuilder(Process):
    """
    Builds reports from
    """

    def __init__(self, process_dir):
        """
        """
        super().__init__(process_dir)

        self.reports_dir = process_dir

    def _run(self, overwrite):
        """
        """
        self._load_reports()

        exam_ids, reports = zip(*self.reports.items())
        df = pd.DataFrame(data=list(reports), index=list(exam_ids))
        df.to_csv(os.path.join(self.reports_dir, "reports.csv"))

    def _load_reports(self):
        """
        Loads the reports from a csv of reports. Reports are stored in the "reports.xlsx"
        file and are labeled by accession number. The file "acc_nums.csv" maps accession
        numbers to exam_ids.
        args:
            reports_dir (str)
        """
        reports_df = pd.read_excel(os.path.join(self.reports_dir, "reports_raw.xlsx"))
        acc_nums_df = pd.read_csv(os.path.join(self.reports_dir, "acc_nums.csv"),
                                  header=None, names=["acc_nums", "exam_dir"],
                                  index_col=0)

        reports = {}
        for acc_num, row in tqdm(acc_nums_df.iterrows(), total=acc_nums_df.shape[0]):
            label, patient_id, exam_id = row["exam_dir"].strip('/').split('/')[-3:]

            # get row with corresponding accession number
            report = reports_df.loc[reports_df['acc_num'] == acc_num]
            if report.shape[0] > 0:
                # takes first report with accession number
                reports[exam_id] = {"description": report["description"].iloc[0],
                                    "report_txt": report["report_txt"].iloc[0],
                                    "classification": report["Classification"].iloc[0]}
        self.reports = reports
