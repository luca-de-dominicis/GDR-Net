import os
import random
import argparse
import json
# The script creates the folder image_set for the epose dataset.
# The folder image_set contains for each class the list of images that belong to that class, and for each split.
parser = argparse.ArgumentParser()

parser.add_argument("--path", help="path to the epose dataset", type=str)

args = parser.parse_args()
assert args.path is not None or args.path != "", "Please specify the path to the epose dataset"

image_set_path = args.path + "/image_set"
images_path = args.path + "rgb/"

# Split the whole list in train and test, with a configurable ratio
def train_test_list(origin_list, ratio=0.2):
    random.shuffle(origin_list)
    split_idx = int(len(origin_list) * ratio)
    train_list = origin_list[split_idx:]
    test_list = origin_list[:split_idx]
    return train_list, test_list

class_names = ["tubetto_m1200", "ugello_l80_99", "dado_m5", "vite_65", "vite_20", "chiave_brugola_6", "deviatore_boccaglio", "chiave_candela_19", "chiave_fissa_8_10", "fascetta_68_73"]
splits = ["all", "train", "test"]

# check for the existence of the directory
if not os.path.exists(image_set_path):
    os.makedirs(image_set_path, exist_ok=True)


lists = []
for i in range(len(class_names)):
    with open(f"{args.path}/test/{str(i + 1).zfill(6)}/scene_gt.json", "r") as f:
        data = json.load(f)
        lists.append(list(data.keys()))

for i in range(len(class_names)):
    train, test = train_test_list(lists[i])
    with open(f"{image_set_path}/{class_names[i]}_train.txt", "w") as f:
        f.write("\n".join(train))
        f.close()
    with open(f"{image_set_path}/{class_names[i]}_test.txt", "w") as f:
        f.write("\n".join(test))
        f.close()
    