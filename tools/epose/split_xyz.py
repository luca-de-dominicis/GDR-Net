import os
from pathlib import Path
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--path", help="path to the epose dataset", type=str)

args = parser.parse_args()
assert args.path is not None and args.path != "", "Please specify the path to the epose dataset"
dataset_path = Path(args.path + "/test")
crop_path = Path(args.path + "/train_pbr" + "/xyz_crop" + "/000000/")

# We define a set with all the image numbers
flat_image_counts = set()

for file in crop_path.glob("*"):
    flat_image_counts.add(file.name.split("_")[0])

flat_list = sorted(list(flat_image_counts))

how_many = {}
how_many["1"] = {}
how_many["2"] = {}
how_many["3"] = {}
how_many["4"] = {}

for i in range(1,5):
    folder = dataset_path / f"00000{i}" / "mask"
    for file in folder.glob("*"):
        if file.name.split("_")[0] in flat_list:
            how_many[str(i)][file.name.split("_")[0]] = how_many[str(i)].get(file.name.split("_")[0], 0) + 1


# Based on how_many we need to copy all instances of each image to the corresponding folder
destination_root = dataset_path / "xyz_crop"

# Create destination folders if they don't exist
for i in range(1, 5):
    (destination_root / f"folder_{i}").mkdir(parents=True, exist_ok=True)

# Function to copy files to the corresponding folder based on how_many
def copy_files_based_on_count():
    arrived = {}
    for image in flat_list:
        arrived[image] = 0
    for i in range(1, 5):  # For each of the 4 folders
        folder = f"folder_{i}"
        for image_number, count in how_many[str(i)].items():
            for j in range(count):
                shutil.copy(crop_path / f"{image_number}_{str(arrived[image_number]).zfill(6)}-xyz.pkl", destination_root / folder / f"{image_number}_{str(count).zfill(6)}-xyz.pkl")
                arrived[image_number] = arrived.get(image_number, 0) + 1


copy_files_based_on_count()