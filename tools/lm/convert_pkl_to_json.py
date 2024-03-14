import pickle
import json
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--path", help="path to the pkl pred", type=str)

args = parser.parse_args()

assert args.path is not None or args.path != "", "Please specify the path to the pkl pred"

# convert pkl to json
with open(args.path, 'rb') as f:
    data = pickle.load(f)

def convert_to_serializable(item):
    """Recursively convert items to a JSON-serializable format."""
    if isinstance(item, torch.Tensor):
        return item.tolist()  # Convert tensors to lists
    elif isinstance(item, np.ndarray):
        return item.tolist()  # Convert NumPy arrays to lists
    elif isinstance(item, dict):
        return {key: convert_to_serializable(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [convert_to_serializable(value) for value in item]
    else:
        return item

# Assuming `data` is your loaded data
serializable_data = convert_to_serializable(data)

# Save the serializable data to a JSON file
with open('output.json', 'w') as f:
    json.dump(serializable_data, f)


