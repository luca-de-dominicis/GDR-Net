import os
import shutil
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def create_directories(base_path):
    dirs_to_create = ["mask", "mask_visib"]
    for dir_name in dirs_to_create:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)

def copy_mask(folders, scene_path):
    index_dict = {}
    for folder in folders:
        mask_folder = folder / "mask"
        mask_folder_dest = scene_path / "mask"
        mask_v_folder = folder / "mask_visib"
        mask_v_folder_dest = scene_path / "mask_visib"

        for img_name in os.listdir(mask_folder):
            img = img_name.split("_")[0]
            index = index_dict.get(img, 0)
            shutil.copy(mask_folder / img_name, mask_folder_dest / f"{img}_{str(index).zfill(6)}.png")
            index_dict[img] = index + 1

        for img_name in os.listdir(mask_v_folder):
            img = img_name.split("_")[0]
            index = index_dict.get(img, 0)
            shutil.copy(mask_v_folder / img_name, mask_v_folder_dest / f"{img}_{str(index).zfill(6)}.png")
            index_dict[img] = index + 1

def merge_json(path, scene_path, info):
    data_list = []
    merged = {}
    for i in range(1,5):
        with open(path / f"00000{i}" / ("scene_gt.json" if not info else "scene_gt_info.json")  , "r") as f:
            data = json.load(f)
            data_list.append(data)
    for key in data_list[0].keys():
        merged[key] = data_list[0][key] + data_list[1][key] + data_list[2][key] + data_list[3][key]
    
    with open(scene_path / ("scene_gt.json" if not info else "scene_gt_info.json"), "w") as f:
        json.dump(merged, f)

parser = argparse.ArgumentParser()

parser.add_argument("--path", help="path to the epose dataset", type=str)

args = parser.parse_args()
assert args.path is not None or args.path != "", "Please specify the path to the epose dataset"
dataset_path = Path(args.path)

if not os.path.exists(dataset_path / "train_pbr"):
    (dataset_path / "train_pbr").mkdir(parents=True, exist_ok=True)
    scene_path = Path(args.path + "/train_pbr/000000/")
    scene_path.mkdir(parents=True, exist_ok=True)
    create_directories(scene_path)

# For the depth it is enough to copy the depth from the first object in the scene
shutil.copytree(dataset_path / "test" / "000001" / "depth", scene_path / "depth")

# Same for the rgbs
shutil.copytree(dataset_path / "test" / "000001" / "rgb", scene_path / "rgb")

# Copy the masks
folders = [dataset_path / "test"/ f"00000{i}" for i in range(1, 5)]
copy_mask(folders, scene_path)

# Merge the json
merge_json(dataset_path / "test", scene_path, False)
merge_json(dataset_path / "test", scene_path, True)

