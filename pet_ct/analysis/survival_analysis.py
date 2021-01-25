import os
from typing import List, Union
import logging
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from lifelines import CoxPHFitter, KaplanMeierFitter

from pet_ct.util.util import process, save_dict_to_json


@process
def load_predictions(
    experiment_dirs: List[str],
    splits="test",
    epoch="best",
    search_subdirs=True,
    process_dir: str = None,
):
    """
    Returns a dict (exams to tasks to list of preds). 
    """
    rows = []
    for experiment_dir in tqdm(experiment_dirs):
        if search_subdirs and os.path.isdir(os.path.join(experiment_dir, "candidates")):
            for dirname in os.listdir(os.path.join(experiment_dir, "candidates")):
                subdir = os.path.join(experiment_dir, "candidates", dirname)
                for split in splits:
                    preds_path = os.path.join(subdir, epoch, f"{split}_preds.csv")
                    preds_df = pd.read_csv(preds_path, index_col=0, header=[0, 1])

                    task = list(preds_df)[0][0]
                    for exam_id, row in preds_df.iterrows():
                        target = int(row[task, "target"].strip("[]"))
                        correct_prob = float(row[task, "correct_prob"].strip("[]"))
                        pred = correct_prob if target == 1 else 1 - correct_prob
                        rows.append({"task": task, "exam_id": exam_id, "pred": pred})

    df = pd.DataFrame(rows)
    if process_dir is not None:
        df.to_csv(os.path.join(process_dir, "preds.csv"))
    return df


TASKS = [
    "carinal_lymph_node",
    "inguinal_lymph_node",
    "left_lungs",
    "cervical_lymph_node",
    "skeleton",
    "chest",
    "head_neck",
    "thoracic_lymph_node",
    "lungs",
    "hilar_lymph_node",
    "pelvis",
    "spine",
    "head",
    "liver",
    "supraclavicular_lymph_node",
    "retroperitoneal_lymph_node",
    "abdominal_lymph_node",
    "mouth",
    "axillary_lymph_node",
    "paratracheal_lymph_node",
    "right_lungs",
    "pelvic_skeleton",
    "iliac_lymph_node",
    "full",
]


def get_indications(
    attrs_df: Union[pd.DataFrame, str],
    mortality_df: Union[pd.DataFrame, str],
    exams_df: Union[pd.DataFrame, str],
    threshold: int = 8,
):
    """
    """
    if isinstance(mortality_df, str):
        mortality_df = pd.read_csv(mortality_df)
    if isinstance(attrs_df, str):
        attrs_df = pd.read_csv(attrs_df, parse_dates=["Study Date"])
    if isinstance(exams_df, str):
        exams_df = pd.read_csv(exams_df)

    df = exams_df[["exam_id", "patient_id"]]
    df = df.merge(
        mortality_df[["days_from_death", "exam_id"]], on="exam_id", how="inner"
    )

    df = df.merge(attrs_df[["exam_id", "Study Description"]], on="exam_id",)

    indication_count = df.groupby("Study Description").size()
    indications = indication_count.index[indication_count >= threshold]
    return list(indications)


# indications with three or more occurence in the test set where the patient
# has a recorded date of death
INDICATIONS = [
    "PET CT BREAST CANCER RESTAGING",
    "PET CT CERVICAL CANCER",
    "PET CT COLORECTAL CANCER",
    "PET CT HEAD NECK CANCER",
    "PET CT LUNG CANCER",
    "PET CT LYMPHOMA",
    "PET CT Ovaries",
    "PET CT UNCOVERED SCAN MISC",
    "PET SCAN DIAGNOSIS BREAST CA",
    "PET SCAN DIAGNOSIS LUNG CANCER",
    "PET SCAN DIAGNOSIS LYMPHOMA",
    "PET SCAN DIAGNOSTIC HEAD NECK CA",
    "PET SCAN REGIONAL OR WHOLE BODY",
    "PET SCAN RESTAGING BREAST CA",
]


ACQUISITION_DATE = datetime(year=2019, month=11, day=4)


@process
def build_new_mortality_df(
    mortality_df: Union[pd.DataFrame, str],
    attrs_df: Union[pd.DataFrame, str],
    exams_df: Union[pd.DataFrame, str],
    split_dir: str = None,
    process_dir=None,
):
    """
    Build a new mortality_df with only the last exam for each patient and using the 
    date of acquisition as the censoring date 
    """
    if isinstance(mortality_df, str):
        mortality_df = pd.read_csv(mortality_df)
    if isinstance(attrs_df, str):
        attrs_df = pd.read_csv(attrs_df, parse_dates=["Study Date"])
    if isinstance(exams_df, str):
        exams_df = pd.read_csv(exams_df)

    # compute `days_from_censor`: the number of days from study to data acquisition
    attrs_df["days_from_followup"] = (ACQUISITION_DATE - attrs_df["Study Date"]).dt.days

    # join the `days_from_censor` on exam_id and patient_id and on "days_from_death"
    df = (
        exams_df[["exam_id", "patient_id"]]
        .merge(attrs_df[["exam_id", "days_from_followup"]], on="exam_id")
        .merge(mortality_df[["exam_id", "days_from_death"]], on=["exam_id"], how="left")
    )

    # only consider last exam per patient
    df = df.loc[df.groupby("patient_id").days_from_followup.idxmin()]

    # double check that all patients are unique
    assert df.patient_id.is_unique
    assert set(exams_df.patient_id) == set(df.patient_id)
    df.to_csv(os.path.join(process_dir, "mortality.csv"), index=False)

    # create new split dir with the subset of exams
    if split_dir is not None:
        new_split_dir = os.path.join(process_dir, "split")
        if not os.path.isdir(new_split_dir):
            os.mkdir(new_split_dir)
        for split in ["train", "test", "valid"]:
            split_df = pd.read_csv(os.path.join(split_dir, f"{split}_exams.csv"))
            split_df = split_df[split_df.exam_id.isin(df.exam_id)]
            split_df.to_csv(os.path.join(new_split_dir, f"{split}_exams.csv"))

    return df


@process
def prepare_survival_data(
    mortality_df: Union[pd.DataFrame, str],
    attrs_df: Union[pd.DataFrame, str],
    preds_df: Union[pd.DataFrame, str],
    exams_df: Union[pd.DataFrame, str],
    indications: List[str] = None,
    process_dir=None,
):

    if isinstance(mortality_df, str):
        mortality_df = pd.read_csv(mortality_df)
    if isinstance(attrs_df, str):
        attrs_df = pd.read_csv(attrs_df, parse_dates=["Study Date"])
    if isinstance(exams_df, str):
        exams_df = pd.read_csv(exams_df)
    if isinstance(preds_df, str):
        preds_df = pd.read_csv(preds_df)

    # consider making this a parameter at some point
    tasks = TASKS
    indications = INDICATIONS if indications is None else indications

    # take median score for each (exam, task) pair
    preds_df = preds_df.groupby(["exam_id", "task"]).pred.median().reset_index()
    # go from one row per (exam, task) to one row per task by creating one column per task
    preds_df = preds_df.pivot(
        index="exam_id", columns="task", values="pred"
    ).reset_index()

    # merge patient_id into attrs_df
    attrs_df = attrs_df.merge(exams_df[["exam_id", "patient_id"]], on="exam_id")

    # drop patients without a patients age
    num_to_drop = attrs_df["Patient's Age"].isna().sum()
    logging.info(f"Dropping {num_to_drop} exams without patient age.")
    attrs_df = attrs_df.dropna(subset=["Patient's Age"]).copy()
    # "Patient's Age" is formatted as "61Y", so drop the "Y"
    attrs_df["patient_age"] = attrs_df["Patient's Age"].apply(
        lambda x: int(x.strip("Y"))
    )

    # set any indication not in indications to "other"
    attrs_df["indication"] = attrs_df["Study Description"].apply(
        lambda x: "other" if x not in indications else x
    )
    # flatten to one-hot with pd.get_dummies and concat to attrs_df
    attrs_df = pd.concat([attrs_df, pd.get_dummies(attrs_df["indication"])], axis=1)

    # prepare dataframe for survival analysis
    df = exams_df[["exam_id", "patient_id", "label"]].rename(
        columns={"label": "summary_code"}
    )
    df = df.merge(preds_df[["exam_id"] + tasks], on="exam_id", how="left")
    df = df.merge(
        mortality_df[["days_from_death", "days_from_followup", "exam_id"]],
        on="exam_id",
        how="left",
    )
    df["event"] = 1 - df.days_from_death.isna()

    # Note: we intentionally exclude "other" so we have n-1 features. This avoids colinearity
    df = df.merge(attrs_df[["exam_id", "patient_age"] + indications], on="exam_id",)
    df["duration"] = df.days_from_death.fillna(df.days_from_followup)

    if process_dir is not None:
        df.to_csv(os.path.join(process_dir, "data.csv"))
    return df


@process
def fit_lr(
    train_df: Union[pd.DataFrame, str],
    covariates: List[str],
    test_df: Union[pd.DataFrame, str] = None,
    threshold: int = 180,
    process_dir: str = None,
):
    if isinstance(train_df, str):
        train_df = pd.read_csv(train_df)
    if isinstance(test_df, str):
        test_df = pd.read_csv(test_df)
    model = LogisticRegression(solver="lbfgs", max_iter=10000)
    included_cols = list(covariates)

    model.fit(train_df[included_cols].values, train_df["duration"] < threshold)

    test_probs = model.predict_proba(test_df[included_cols].values)
    results = {
        "auroc": roc_auc_score(test_df["duration"] < threshold, test_probs[:, -1])
    }

    if process_dir is not None:
        save_dict_to_json(os.path.join(process_dir, "results.json"), results)

    return results


@process
def fit_cox(
    train_df: Union[pd.DataFrame, str],
    covariates: List[str],
    test_df: Union[pd.DataFrame, str] = None,
    strata: List[str] = None,
    plot: bool = False,
    process_dir: str = None,
):
    if isinstance(train_df, str):
        train_df = pd.read_csv(train_df)
    if isinstance(test_df, str):
        test_df = pd.read_csv(test_df)
    cphf = CoxPHFitter()
    included_cols = ["duration", "event"] + list(covariates)
    print(train_df.columns)
    cphf.fit(
        train_df[included_cols],
        duration_col="duration",
        event_col="event",
        strata=strata,
    )

    results = {
        "log_likelihood": cphf.log_likelihood_,
        "concordance_index": cphf.concordance_index_,
        "log_likelihood_ratio_test_pvalue": cphf.log_likelihood_ratio_test().p_value,
    }

    if test_df is not None:
        results["test_log_likelihood"] = cphf.score(
            test_df[included_cols], scoring_method="log_likelihood"
        )
        results["test_concordance_index"] = cphf.score(
            test_df[included_cols], scoring_method="concordance_index"
        )

    if plot and process_dir is not None:
        plt.figure(figsize=(5, 10))
        cphf.plot()
        plt.savefig(os.path.join(process_dir, "hazard_plot.pdf"))

    if process_dir is not None:
        cphf.summary.to_csv(os.path.join(process_dir, "summary.csv"))
        save_dict_to_json(os.path.join(process_dir, "results.json"), results)

    return results, cphf


@process
def fit_univariable_coxs(
    train_df: Union[pd.DataFrame, str],
    covariates: List[str],
    strata: List[str] = None,
    process_dir=None,
):
    summary = []
    for covariate in covariates:
        results, cphf = fit_cox(
            train_df=train_df, covariates=[covariate], strata=strata, process_dir=None,
        )
        summary.append(cphf.summary.reset_index().iloc[0].to_dict())
    summary = pd.DataFrame(summary)
    if process_dir is not None:
        summary.to_csv(os.path.join(process_dir, "summary.csv"))
    return summary


def cox_likelihood_ratio_test(
    train_df: Union[pd.DataFrame, str],
    covariates: List[str],
    restricted_covariates: List[str],
):
    covariates, restricted_covariates = set(covariates), set(restricted_covariates)
    if not restricted_covariates.issubset(covariates):
        raise ValueError(f"`restricted_covariates` must be a subset of `covariates`")

    restricted_cox_results, _ = fit_cox(train_df, (covariates - restricted_covariates))
    cox_results, _ = fit_cox(train_df, covariates)

    pvalue = chi2.sf(
        2
        * (cox_results["log_likelihood"] - (restricted_cox_results["log_likelihood"])),
        df=len(restricted_covariates),
    )
    return pvalue


def cox_likelihood_ratio_test(
    train_df: Union[pd.DataFrame, str],
    covariates: List[str],
    restricted_covariates: List[str],
):
    covariates, restricted_covariates = set(covariates), set(restricted_covariates)
    if not restricted_covariates.issubset(covariates):
        raise ValueError(f"`restricted_covariates` must be a subset of `covariates`")

    restricted_cox_results, _ = fit_cox(train_df, (covariates - restricted_covariates))
    cox_results, _ = fit_cox(train_df, covariates)

    pvalue = chi2.sf(
        2
        * (cox_results["log_likelihood"] - (restricted_cox_results["log_likelihood"])),
        df=len(restricted_covariates),
    )
    return pvalue


@process
def kaplan_meier_curve(
    data_df: Union[pd.DataFrame, str],
    task: str = "liver",
    threshold: Union[float, List] = 0.5,
    process_dir: str = None,
):
    if isinstance(data_df, str):
        data_df = pd.read_csv(data_df)

    if isinstance(threshold, float):
        thresholds = [threshold, 1]
    else:
        thresholds = threshold
        thresholds.append(1)

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    prev_threshold = -1
    for threshold in thresholds:
        name = f"{task}: {prev_threshold} < y <= {threshold}"
        grouped_df = data_df[
            (data_df[task] > prev_threshold) & (data_df[task] <= threshold)
        ]

        kmf.fit(grouped_df["duration"], grouped_df["event"], label=name)
        kmf.plot(ax=ax)
        prev_threshold = threshold

    plt.xlabel("Follow-up time (days)")
    plt.ylabel("Probability of survival")

    if process_dir is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(process_dir, f"{task}_kaplan_meier.pdf"))
