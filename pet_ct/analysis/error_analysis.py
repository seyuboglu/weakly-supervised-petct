"""
helpful scripts to go with <EXPERIMENT_PATH>/notebook.ipynb or error_analysis.ipynb.
"""
import os

import pandas as pd

from pet_ct.learn.dataloaders import mt_mi_exam_collate, mt_exam_collate, exam_collate


def get_exam_info(exam_id, task, preds_df, dataset, multitask=False, multiinput=False):
    """
    """
    target = int(preds_df.loc[exam_id][(task, 'target')][1:-1])
    pred = int(preds_df.loc[exam_id][(task, 'pred')][1:-1])
    correct_prob = float(preds_df.loc[exam_id][(task, 'correct_prob')][1:-1])
    inputs, targets, info = dataset.get_exam(exam_id)
    if multitask:
        if multiinput:
            inputs, targets, info = mt_mi_exam_collate([(inputs, targets, info)])
        else:
            inputs, targets, info = mt_exam_collate([(inputs, targets, info)])
    else:
        inputs, targets, info = exam_collate([(inputs, targets, info)])
    num_imgs_used = len(inputs[0])
    patient_id = info[0]['patient_id']
    return {
        "patient_id": patient_id,
        "exam_id": exam_id,
        "target": target,
        "pred": pred,
        "correct_prob": correct_prob,
        "inputs": inputs,
        "targets": targets,
    }


def get_image_count_from_source(patient_id, exam_id):
    """
    """
    for level in [1, 2, 4, 9]:
        curr_dir = f'/data4/data/fdg-pet-ct/exams/{level}/{patient_id}/{exam_id}/'
        if os.path.isdir(curr_dir):
            num_imgs_pet = len(os.listdir(os.path.join(curr_dir, 'PET_BODY_CTAC')))
            num_imgs_ct = len(os.listdir(os.path.join(curr_dir, 'CT Images')))
            exam_dir = curr_dir
            break
    return {
        "source_dir": curr_dir,
        "num_imgs_pet": num_imgs_pet,
        "num_imgs_ct": num_imgs_ct
    }


def get_relevant_exam_ids(preds_df, task, result_type=None):
    """
    gets prediction dicts based on result_type
    """
    if result_type and \
       result_type not in {'TP', 'FP', 'TN', 'FN', 'T', 'F'}:
        raise ValueError(
            "result_type must be one of " +
            "{'TP', 'FP', 'TN', 'FN', 'T', 'F'}"
        )
    exam_ids = []
    for exam_id in preds_df.index[1:]:
        row = dict(preds_df.loc[exam_id])

        correct_prob = float(row[(task, 'correct_prob')][1:-1])
        target = int(row[(task, 'target')][1:-1])
        pred = int(row[(task, 'pred')][1:-1])

        if not result_type:
            exam_ids.append(exam_id)
        if result_type:
            if result_type_match(pred, target, result_type):
                exam_ids.append(exam_id)
    return exam_ids


def result_type_match(pred, target, result_type):
    """
    Verifies nature of prediction and target given a result_type.
    """
    if result_type == 'T' and pred == target:
        return True
    if result_type == 'F' and pred != target:
        return True
    if result_type == 'TP' and pred == target and target == 1:
        return True
    if result_type == 'FP' and pred != target and target == 0:
        return True
    if result_type == 'TN' and pred == target and target == 0:
        return True
    if result_type == 'FN' and pred != target and target == 1:
        return True

    return False
