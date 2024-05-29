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

print("Collecting image numbers...")
for file in crop_path.glob("*"):
    flat_image_counts.add(file.name.split("_")[0])

flat_list = sorted(list(flat_image_counts))
idx2class = {
    1: "tubetto_m1200",
    2: "ugello_l80_90",
    3: "dado_m5",
    4: "vite_65",
    5: "vite_20",
    6: "chiave_brugola_6",
    7: "deviatore_boccaglio",
    8: "chiave_candela_19",
    9: "chiave_fissa_8_10",
    10: "fascetta_68_73",
}
how_many = {}
for i in range(1,len(idx2class.keys()) + 1):
    how_many[str(i)] = {}

print(how_many)

print("Counting instances of each image...")
for i in range(1, len(idx2class.keys()) + 1):
    folder = dataset_path / str(i).zfill(6) / "mask"
    for file in folder.glob("*"):
        if file.name.split("_")[0] in flat_list:
            how_many[str(i)][file.name.split("_")[0]] = how_many[str(i)].get(file.name.split("_")[0], 0) + 1


# Based on how_many we need to copy all instances of each image to the corresponding folder
destination_root = dataset_path / "xyz_crop"

print("Copying files...")
# Create destination folders if they don't exist
for i in range(1, len(idx2class.keys()) + 1):
    (destination_root / f"{str(i).zfill(6)}").mkdir(parents=True, exist_ok=True)

# Function to copy files to the corresponding folder based on how_many
def copy_files_based_on_count():
    arrived = {}
    for image in flat_list:
        arrived[image] = 0
    for i in range(1, len(idx2class.keys()) + 1):  # For each of the 4 folders
        folder = f"{str(i).zfill(6)}"
        print(i)
        for image_number, count in how_many[str(i)].items():
            for j in range(count):
                shutil.copy(crop_path / f"{image_number}_{str(arrived[image_number]).zfill(6)}-xyz.pkl", destination_root / folder / f"{image_number}_{str(j).zfill(6)}-xyz.pkl")
                arrived[image_number] = arrived.get(image_number, 0) + 1


copy_files_based_on_count()