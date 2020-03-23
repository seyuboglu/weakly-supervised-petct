"""
"""
import re

def get_left_region(report, pattern):
    rem_string = report[:]
    offset = 0
    regex = re.compile(r'\b(left)' +
                       r'(( (\w+ ){0,2}\w+,)*? (\w+ ){0,2}?)' +
                       f'({pattern})' +
                       r'\b')
    while True:
        match = re.search(regex, rem_string)
        if not match:
            return None

        match_str = match.group(0)
        left_idx = match_str.rfind('left')

        true_match = True
        for nonmatch_keyword in ['right', 'bilateral']:
            nonmatch_idx = match_str.rfind(nonmatch_keyword)
            if (left_idx == -1) or (left_idx < nonmatch_idx):
                offset = match.end()
                rem_string = rem_string[offset:]
                true_match = False
                break
        if true_match:
            break
        if len(rem_string) == 0:
            return None

    real_idx = match_str.rfind(f'left')
    if real_idx == -1:
        real_idx = match_str.rfind(pattern)

    return {
        "start": offset + match.start() + real_idx,
        "end": offset + match.end()
    }

def get_right_region(report, pattern):
    rem_string = report[:]
    offset = 0
    regex = re.compile(r'\b(right)' +
                       r'(( (\w+ ){0,2}\w+,)*? (\w+ ){0,2}?)' +
                       f'({pattern})' +
                       r'\b')
    while True:
        match = re.search(regex, rem_string)
        if not match:
            return None

        match_str = match.group(0)
        right_idx = match_str.rfind('right')

        true_match = True
        for nonmatch_keyword in ['left', 'bilateral']:
            nonmatch_idx = match_str.rfind(nonmatch_keyword)
            if (right_idx == -1) or (right_idx < nonmatch_idx):
                offset = match.end()
                rem_string = rem_string[offset:]
                true_match = False
                break
        if true_match:
            break
        if len(rem_string) == 0:
            return None

    real_idx = match_str.rfind(f'right {pattern}')
    if real_idx == -1:
        real_idx = match_str.rfind(pattern)

    return {
        "start": offset + match.start() + real_idx,
        "end": offset + match.end()
    }

def get_upper_region(report, pattern):
    rem_string = report[:]
    offset = 0
    regex = re.compile(r'\b(upper)' +
                       r'(( (\w+ ){0,2}\w+,)*? (\w+ ){0,2}?)' +
                       f'({pattern})' +
                       r'\b')
    while True:
        match = re.search(regex, rem_string)
        if not match:
            return None

        match_str = match.group(0)
        upper_idx = match_str.rfind('upper')

        true_match = True
        for nonmatch_keyword in ['lower']:
            nonmatch_idx = match_str.rfind(nonmatch_keyword)
            if (upper_idx == -1) or (upper_idx < nonmatch_idx):
                offset = match.end()
                rem_string = rem_string[offset:]
                true_match = False
                break
        if true_match:
            break
        if len(rem_string) == 0:
            return None

    real_idx = match_str.rfind(f'upper {pattern}')
    if real_idx == -1:
        real_idx = match_str.rfind(pattern)

    return {
        "start": offset + match.start() + real_idx,
        "end": offset + match.end()
    }

def get_lower_region(report, pattern):
    rem_string = report[:]
    offset = 0
    regex = re.compile(r'\b(lower)' +
                       r'(( (\w+ ){0,2}\w+,)*? (\w+ ){0,2}?)' +
                       f'({pattern})' +
                       r'\b')
    while True:
        match = re.search(regex, rem_string)
        if not match:
            return None

        match_str = match.group(0)
        lower_idx = match_str.rfind('lower')

        true_match = True
        for nonmatch_keyword in ['upper']:
            nonmatch_idx = match_str.rfind(nonmatch_keyword)
            if (lower_idx == -1) or (lower_idx < nonmatch_idx):
                offset = match.end()
                rem_string = rem_string[offset:]
                true_match = False
                break
        if true_match:
            break
        if len(rem_string) == 0:
            return None

    real_idx = match_str.rfind(f'lower {pattern}')
    if real_idx == -1:
        real_idx = match_str.rfind(pattern)

    return {
        "start": offset + match.start() + real_idx,
        "end": offset + match.end()
    }