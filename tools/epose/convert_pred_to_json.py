import pickle
import json
import numpy as np
import torch
import argparse

def convert_to_serializable(item):
    """
    Recursively convert complex items like tensors into lists or dicts so they can be serialized to JSON.
    """
    if isinstance(item, dict):
        return {key: convert_to_serializable(value) for key, value in item.items()}
    elif hasattr(item, 'tolist'):  # This checks if the item is a tensor
        return item.tolist()  # Convert tensors to lists
    elif isinstance(item, (list, tuple)):
        return [convert_to_serializable(sub_item) for sub_item in item]
    else:
        return item

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Predictions path")
    args = parser.parse_args()

    assert args.path != None and args.path != '', 'Please provide the path to the predictions'

    with open(args.path, "rb") as f:
        preds_pkl = pickle.load(f)

    
    # Convert the original OrderedDict to a serializable format
    serializable_dict = {key: convert_to_serializable(dict(value)) for key, value in preds_pkl.items()}


    with open(args.path.split('.pkl')[0] + '.json', 'w') as f:
        json.dump(serializable_dict, f)

    