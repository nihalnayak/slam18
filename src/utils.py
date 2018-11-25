"""This program contains utilities that are used by other prog
"""
import json


def convert_to_bool(value):
    """Function to convert string value to boolean
    Args:
        value (str): String value of the boolean value

    Returns:
        bool: True/False
    """
    if value.lower() == 'true':
        return True
    return False


def load_context_json(_file):
    # the name of the json file will be the same with 
    with open(_file + ".json") as fp:
        data = json.load(fp)        
    return data


def load_labels(filename):
    """
    This loads labels, either the actual ones or your predictions.

    Parameters:
        filename: the filename pointing to your labels

    Returns:
        labels: a dict of instance_ids as keys and labels between 0 and 1 as values
    """
    labels = dict()

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                line = line.split()
            instance_id = line[0]
            label = float(line[1])
            labels[instance_id] = label
    return labels
