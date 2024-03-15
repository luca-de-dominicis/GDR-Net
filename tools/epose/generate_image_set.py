import os
import random
import argparse

# The script creates the folder image_set for the epose dataset.
# The folder image_set contains for each class the list of images that belong to that class, and for each split.
parser = argparse.ArgumentParser()

parser.add_argument("--path", help="path to the epose dataset", type=str)

args = parser.parse_args()
assert args.path is not None or args.path != "", "Please specify the path to the epose dataset"

image_set_path = args.path + "/image_set"
images_path = args.path + "/test/000001/rgb/"

# Split the whole list in train and test, with a configurable ratio
def train_test_list(origin_list, ratio=0.2):
    random.shuffle(origin_list)
    split_idx = int(len(origin_list) * ratio)
    train_list = origin_list[split_idx:]
    test_list = origin_list[:split_idx]
    return train_list, test_list

class_names = ["chiave_candela_19", "ugello_l80_90", "dado_m5", "vite_65"]
splits = ["all", "train", "test"]

# check for the existence of the directory
if not os.path.exists("image_set"):
    os.mkdir(image_set_path)


original_list = os.listdir(images_path) # lists all the images in the dataset
original_list = [name[:-4] for name in original_list]
train_list, test_list = train_test_list(original_list, ratio=0.2)
train_list = sorted(train_list)
test_list = sorted(test_list)

for class_name in class_names:
    for split in splits:
        with open(f"{image_set_path}/{class_name}_{split}.txt", "w") as f:
            if split == "all":
                f.write("\n".join(original_list))
            elif split == "train":
                f.write("\n".join(train_list))
            elif split == "test":
                f.write("\n".join(test_list))
            f.close()
        