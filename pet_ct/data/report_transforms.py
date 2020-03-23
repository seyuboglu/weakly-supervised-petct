"""
"""
import re
import random

import nltk

import numpy as np


def to_lower(impression):
    """
    """
    return impression.lower()


def strip(impression, chars=" "):
    """
    """
    return impression.strip(chars)


def word_tokenize(report):
    """
    """
    return nltk.word_tokenize(report)

def sent_tokenize(report):
    """
    """
    return nltk.sent_tokenize(report)

def split_impression_sections(impression):
    """
    """
    sections = re.findall("[1-9]\. (.+?(?=[1-9]\. |$))", impression)
    if not sections:
        sections = [impression]

    # remove empty sections
    sections = [section for section in sections if len(nltk.word_tokenize(section)) > 0]

    # if no more sections remain
    if not sections:
        sections = ["none"]

    return sections


def sample_impression_sections(impression, distribution="uniform"):
    """
    """
    sections = split_impression_sections(impression)
    if distribution == "uniform":
        section = random.choices(sections)[0]
    elif distribution == "decreasing":
        section = random.choices(sections,
                                 weights=[1 / (i + 1) for i in range(len(sections))])
    elif distribution == "log_decreasing":
        section = random.choices(sections,
                                 weights=[1 / (np.log(i + 1) + 1) for i in range(len(sections))])
    else:
        raise ValueError()

    return section


def extract_impression(report):
    """
    """
    report = re.sub("(HISTOLOGY|PET SEQUENCE|COMPARISON|SEQUENCE|PET sequence|Histology|PET CT):[ ]*[^ ]*\.", "", report)
    # remove dates
    report = re.sub("[0-9]*\/[0-9]*\/[0-9]*", "", report)

    match = re.findall("IMPRESSION:(.*)END OF IMPRESSION", report)

    if not match:
        match = re.findall("IMPRESSION:(.*)SUMMARY", report)

    if not match:
        match = re.findall("IMPRESSION:.* (1\.[ ]*[^a-z]*)", report)

    if not match:
        match = re.findall("IMPRESSION:[ ]* ([^a-z]*)", report)

    if not match:
        return ""

    impression_text = match[0]

    if len(impression_text) < 10:
        match = re.search("[1-9]\. [^a-z]*", report)
        if match is not None:
            impression_text += match.group()

    # remove trailing capital letters
    impression_text = re.sub(".[ ]*[^a-z | x ]*$", "", impression_text)

    return impression_text
